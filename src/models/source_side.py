"""Train-only preprocessing; validation-frozen thresholds; parallel anomaly/classifier."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from src.features.source_side import extract_source_features
from src.evaluation.source_side import alarm_policy, event_metrics, summarize_events
from src.simulator.source_side import load_source_config


class SourceMonitor:
    schema_version=2

    def __init__(self,profile='minimal',temporal=True,environment=True,seed=42,config=None):
        self.profile=profile; self.temporal=temporal; self.environment=environment
        self.seed=seed; self.config=config or load_source_config(); self.thresholds={}

    def features(self,session):
        return extract_source_features(session.observations,self.profile,self.temporal,self.environment,self.config)

    def fit(self,train,validation):
        train_ids={s.metadata['device_id'] for s in train}
        val_ids={s.metadata['device_id'] for s in validation}
        if train_ids & val_ids: raise ValueError('device leakage')
        self.train_devices=sorted(train_ids); self.validation_devices=sorted(val_ids)
        frames=[self.features(s) for s in train]
        self.feature_names=list(frames[0])
        x=np.concatenate([f.to_numpy() for f in frames]); y=np.concatenate([s.truth.family.to_numpy() for s in train])
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        self.scaler=RobustScaler()
        z=self.scaler.fit_transform(self.imputer.fit_transform(x))
        self.classes=np.unique(y)
        self.classifier=XGBClassifier(n_estimators=self.config['monitor']['model_estimators'],max_depth=4,
            learning_rate=.07,subsample=.85,colsample_bytree=.85,random_state=self.seed,n_jobs=2)
        self.classifier.fit(z,np.searchsorted(self.classes,y))
        # Only explicitly benign calibration sessions supply the normal model.
        benign=np.concatenate([np.full(len(s.observations),s.metadata['scenario']==0) for s in train])
        self.anomaly=IsolationForest(n_estimators=80,random_state=self.seed,n_jobs=2).fit(z[benign])
        self.centres=np.median(z[benign],axis=0)
        self.scales=np.maximum(np.quantile(abs(z[benign]-self.centres),.9,axis=0),.1)
        self.value_indices=[i for i,f in enumerate(self.feature_names) if f.endswith('__value')]
        receiver = np.concatenate([self.receiver_values(s) for s in train if s.metadata['scenario']==0])
        self.receiver_center = np.nanmedian(receiver,axis=0)
        self.receiver_scale = np.maximum(np.nanquantile(abs(receiver-self.receiver_center),.9,axis=0),1e-8)
        # Disjoint validation devices: probability/score calibration versus threshold selection.
        if len(val_ids)<2:
            raise ValueError('at least two validation devices required')
        calibration_ids=set(sorted(val_ids)[::2])
        calibration=[s for s in validation if s.metadata['device_id'] in calibration_ids]
        selection=[s for s in validation if s.metadata['device_id'] not in calibration_ids]
        self.calibration_devices=sorted(calibration_ids)
        self.selection_devices=sorted(val_ids-calibration_ids)
        self.temperature=1.
        probs=np.concatenate([self.classifier.predict_proba(self.transform(s)) for s in calibration])
        targets=np.concatenate([s.truth.family.to_numpy() for s in calibration])
        if not set(targets).issubset(set(self.classes)):
            raise ValueError('calibration contains held-out mechanism labels')
        target_indices=np.searchsorted(self.classes,targets)
        from scipy.optimize import minimize_scalar
        def objective(log_temperature):
            p=self.calibrate_probability(probs,np.exp(log_temperature))
            return -np.log(np.maximum(p[np.arange(len(p)),target_indices],1e-12)).mean()
        fit=minimize_scalar(objective,bounds=(-2,2),method='bounded')
        self.temperature=float(np.exp(fit.x))
        calibrated=self.calibrate_probability(probs,self.temperature)
        correct=calibrated.argmax(1)==target_indices
        confidence=calibrated.max(1)[correct]
        self.unknown_cutoff=float(np.quantile(confidence,self.config['monitor']['unknown_confidence_quantile'])) if len(confidence) else 1.
        self.score_references={}
        normal_scores=[self.raw_scores(s) for s in calibration if s.metadata['scenario']==0]
        for name in ('classifier','anomaly'):
            self.score_references[name]=np.sort(np.concatenate([v[name] for v in normal_scores]))
        cached=[(s,self.raw_scores(s)) for s in selection]
        budget=self.config['monitor']['false_alarm_budget_per_hour']
        self.validation_operating_points={}
        for name in ('limits','ewma','cusum','anomaly','classifier','combined','receiver'):
            vals=np.concatenate([v[name] for s,v in cached if s.metadata['scenario']==0])
            candidates=np.unique(np.r_[np.quantile(vals,np.linspace(.5,1,35)),vals.max()+1e-6])
            feasible=[]
            for threshold in candidates:
                rows=[event_metrics(s,self.policy(v[name],threshold)[0]) for s,v in cached]
                metric=summarize_events(rows)
                if metric['false_alarms_per_hour']<=budget:
                    feasible.append((metric['detection_rate'], -threshold, threshold,metric))
            # Largest detection on validation; ties prefer the lower threshold.
            _,_,threshold,metric=max(feasible,key=lambda r:r[:2])
            self.thresholds[name]=float(threshold)
            self.validation_operating_points[name]=metric
        return self

    @staticmethod
    def calibrate_probability(probabilities,temperature):
        logits=np.log(np.maximum(probabilities,1e-12))/temperature
        exp=np.exp(logits-logits.max(axis=1,keepdims=True))
        return exp/exp.sum(axis=1,keepdims=True)

    @staticmethod
    def receiver_values(session):
        return session.observations[['protocol_gain','protocol_qber']].rolling(5,min_periods=1).mean().to_numpy()

    @staticmethod
    def normalized_excursion(values, reference):
        """Median-to-95th-percentile units, with unbounded upper tail.

        Unlike a finite empirical CDF, this retains ordering beyond the largest
        calibration observation. It is an excursion score, not a probability.
        """
        middle, upper=np.quantile(reference,[.5,.95])
        return (values-middle)/max(float(upper-middle),1e-8)

    def transform(self,session):
        f=self.features(session)
        if list(f)!=self.feature_names: raise ValueError('checkpoint feature schema mismatch')
        return self.scaler.transform(self.imputer.transform(f.to_numpy()))

    def policy(self,score,threshold):
        m=self.config['monitor']
        return alarm_policy(score,threshold,m['persistence'],m['clear_samples'])

    def raw_scores(self,session):
        z=self.transform(session)
        p=self.calibrate_probability(self.classifier.predict_proba(z),getattr(self,'temperature',1.))
        normal=int(np.flatnonzero(self.classes==0)[0])
        classifier=1-p[:,normal]
        anomaly=-self.anomaly.score_samples(z)
        residual=np.max(abs((z[:,self.value_indices]-self.centres[self.value_indices])/self.scales[self.value_indices]),axis=1)
        ewma=[]; cusum=[]; e=u=0.
        for r in residual:
            e=.15*r+.85*e; u=max(0.,u+r-1.5)
            ewma.append(e); cusum.append(u)
        limits=np.zeros(len(z)); violations={}
        for k,(lo,hi) in self.config['engineering_limits'].items():
            if k not in session.observations: continue
            v=session.observations[k].to_numpy(); width=hi-lo
            exceed=np.maximum((lo-v)/width,(v-hi)/width)
            violations[k]=np.nan_to_num(np.maximum(exceed,0)).tolist()
            limits=np.maximum(limits,np.nan_to_num(np.maximum(exceed,0)))
        # Benign calibration excursions put both branches on a common scale.
        # Max is a union score, NOT an attack probability. Its own validation cutoff
        # accounts for multiplicity/correlation; neither branch gates the other.
        ranks={}
        for name,values in [('classifier',classifier),('anomaly',anomaly)]:
            reference=getattr(self,'score_references',{}).get(name)
            ranks[name]=self.normalized_excursion(values,reference) if reference is not None else np.zeros(len(z))
        receiver=np.nan_to_num(abs((self.receiver_values(session)-self.receiver_center)/self.receiver_scale),nan=0.).max(axis=1)
        return dict(limits=limits,ewma=np.array(ewma),cusum=np.array(cusum),anomaly=anomaly,receiver=receiver,
                    classifier=classifier,combined=np.maximum(ranks['classifier'],ranks['anomaly']),
                    classifier_excursion=ranks['classifier'],anomaly_excursion=ranks['anomaly'],
                    probabilities=p,violations=violations)

    def predict(self,session):
        scores=self.raw_scores(session)
        alarm,warning=self.policy(scores['combined'],self.thresholds['combined'])
        labels=self.classes[scores['probabilities'].argmax(1)]
        # Unknown is an operational rejection, not proof of an unseen mechanism.
        anomaly_alarm,_=self.policy(scores['anomaly'],self.thresholds['anomaly'])
        uncertain=anomaly_alarm & ((labels==0)|(scores['probabilities'].max(1)<self.unknown_cutoff))
        health_cols=[k for k in session.observations if k.endswith(('__missing','__saturated','__stale'))]
        health=session.observations[health_cols].max(axis=1).to_numpy()>0
        return dict(scores={k:v.tolist() for k,v in scores.items() if k not in ('probabilities','violations')},
                    probabilities=scores['probabilities'].tolist(),classes=self.classes.tolist(),
                    family=labels.tolist(),alarm=alarm.tolist(),warning=warning.tolist(),
                    uncertain=uncertain.tolist(),sensor_fault=health.tolist(),violations=scores['violations'],
                    responsible_branch=np.where(scores['classifier_excursion']>=scores['anomaly_excursion'],
                                                'classifier','anomaly').tolist())

    def explain(self,session,index):
        import shap
        z=self.transform(session)[index:index+1]
        if not hasattr(self,'_explainer'):
            self._explainer=shap.TreeExplainer(self.classifier)
        values=np.asarray(self._explainer.shap_values(z))
        predicted=int(self.classifier.predict(z)[0])
        v=values[0,:,predicted] if values.ndim==3 else values[0]
        order=np.argsort(abs(v))[::-1][:12]
        return [dict(feature=self.feature_names[i],contribution=float(v[i])) for i in order]
