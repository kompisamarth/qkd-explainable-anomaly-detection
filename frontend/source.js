/* Source telemetry UI. Truth is fetched only on explicit reveal. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  let session=null, meta=null, generation=0;
  const text=(id,value)=>{$(id).textContent=value;};
  async function api(path,options){
    const response=await fetch('/api/source/'+path,options);
    const data=await response.json();
    if(!response.ok) throw Error(data.detail||response.statusText);
    return data;
  }
  function controls(){
    $('run').disabled=!meta?.model_available[$('profile').value];
    $('reveal').disabled=!session?.blind || !!session?.truth;
    $('explain').disabled=!session;
    $('sample').disabled=!session;
    for(const id of ['scenario','seed']) $(id).disabled=$('blind').checked;
  }
  function clearSession(){
    session=null; generation++;
    for(const id of ['charts','shap','detail','summary','provenance','sampleLabel']) $(id).replaceChildren();
    $('timeline').getContext('2d').clearRect(0,0,1000,220);
    text('truth','No current session.'); controls();
  }
  function plot(canvas,values,low=null,high=null){
    canvas.width=Math.max(220,Math.round(canvas.clientWidth));
    const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    ctx.clearRect(0,0,w,h);
    const finite=values.filter(Number.isFinite);
    if(!finite.length){ctx.fillText('No valid readings',50,50);return;}
    let lo=Math.min(...finite),hi=Math.max(...finite);
    if(low!==null)lo=Math.min(lo,low); if(high!==null)hi=Math.max(hi,high);
    const y=v=>h-45-(v-lo)/(hi-lo||1)*(h-70),x=i=>55+i/(values.length-1)*(w-70);
    ctx.font='12px sans-serif';ctx.fillStyle='#475569';
    ctx.fillText(hi.toPrecision(3),0,20);ctx.fillText(lo.toPrecision(3),0,h-45);
    [0,.5,1].forEach(f=>{const i=Math.round(f*(values.length-1));ctx.fillText(session.observations[i].time_s+' s',x(i)-12,h-8);});
    for(const limit of [low,high])if(limit!==null){
      ctx.strokeStyle='#b45309';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(55,y(limit));ctx.lineTo(w-15,y(limit));ctx.stroke();ctx.setLineDash([]);
    }
    ctx.strokeStyle='#0d9488';ctx.lineWidth=2;ctx.beginPath();let pen=false;
    values.forEach((v,i)=>{if(!Number.isFinite(v)){pen=false;return;}if(pen)ctx.lineTo(x(i),y(v));else ctx.moveTo(x(i),y(v));pen=true;});ctx.stroke();
  }
  function timeline(){
    const canvas=$('timeline');plot(canvas,session.prediction.scores.combined,session.thresholds.combined);
    const ctx=canvas.getContext('2d'),n=session.observations.length,span=canvas.width-70;
    session.prediction.alarm.forEach((v,i)=>{if(v){ctx.fillStyle='#dc2626';ctx.fillRect(55+i/(n-1)*span,180,Math.max(2,span/n),5);}});
    if(session.truth){
      session.truth.truth.forEach((row,i)=>{
        if(row.active){ctx.fillStyle=row.requested_mechanism===4?'#7c3aed':'#0284c7';ctx.fillRect(55+i/(n-1)*span,187,Math.max(2,span/n),5);}
        if(row.degraded){ctx.fillStyle='#d97706';ctx.fillRect(55+i/(n-1)*span,194,Math.max(2,span/n),5);}
      });
      text('truth',JSON.stringify({schedule:session.truth.metadata.schedule,metrics:session.truth.metrics},null,2));
    }
    text('summary','Validation-normalized union score (not a probability). Red: alarm. Revealed truth: purple damage exposure, blue other exposure, amber persistent/component effects.');
  }
  function inspect(){
    if(!session)return;
    const i=+$('sample').value,p=session.prediction;
    text('sampleLabel',session.observations[i].time_s+' s');
    text('detail',JSON.stringify({family:meta.families[p.family[i]],probabilities:p.probabilities[i],
      alarm:p.alarm[i],alarm_origin:p.alarm_origin[i],dominant_score_branch:p.responsible_branch[i],warning:p.warning[i],
      uncertain:p.uncertain[i],sensor_fault:p.sensor_fault[i]},null,2));
  }
  function render(){
    const obs=session.observations;$('charts').replaceChildren();
    Object.keys(session.limits).filter(k=>k in obs[0]).forEach(k=>{
      const card=document.createElement('div');card.className='card';
      const title=document.createElement('h3'),caption=document.createElement('p'),canvas=document.createElement('canvas');
      canvas.width=600;canvas.height=220;canvas.setAttribute('role','img');
      const relative=k==='emitted_intensity';
      title.textContent=relative?'Emitted / nominal intensity (ratio)':k+' ('+session.units[k]+')';
      canvas.setAttribute('aria-label',title.textContent+' over physical time');
      caption.textContent='Dashed: training-benign median ± 90th-percentile absolute deviation; not a security limit.';
      card.append(title,canvas,caption);$('charts').append(card);
      plot(canvas,obs.map(r=>r[k]===null?null:relative?r[k]/r.nominal_intensity:r[k]),...session.bands[k]);
    });
    timeline();$('sample').max=obs.length-1;$('sample').value=obs.length-1;inspect();
    if(!session.truth)text('truth','Hidden until reveal.');
    text('provenance',JSON.stringify(session.provenance));controls();
  }
  $('run').onclick=async()=>{
    clearSession();const request=generation;text('status','Simulating…');$('run').disabled=true;
    try{
      const result=await api('simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
        profile:$('profile').value,scenario:+$('scenario').value,seed:+$('seed').value,blind:$('blind').checked,stress:$('stress').value})});
      if(request!==generation)return;session=result;render();text('status','Simulation ready. A normal prediction does not certify security.');
    }catch(error){if(request===generation){clearSession();text('status','Simulation failed: '+error.message);}}
    finally{controls();}
  };
  $('reveal').onclick=async()=>{
    if(!session)return;const sid=session.session_id;
    try{const result=await api('reveal/'+sid,{method:'POST'});if(session?.session_id!==sid)return;session.truth=result;timeline();controls();}
    catch(error){clearSession();text('status',error.message);}
  };
  $('explain').onclick=async()=>{
    if(!session)return;const sid=session.session_id,index=$('sample').value;
    try{
      const result=await api('explain/'+sid+'?index='+index);
      if(session?.session_id!==sid || $('sample').value!==index)return;
      $('shap').replaceChildren();
      result.attributions.forEach(a=>{const p=document.createElement('p');p.textContent=a.feature+': '+a.contribution.toFixed(4);$('shap').append(p);});
      text('status',result.note+' Dominant fusion branch: '+result.responsible_branch+'.');
    }catch(error){if(session?.session_id===sid){clearSession();text('status',error.message);}}
  };
  $('sample').oninput=()=>{inspect();$('shap').replaceChildren();};
  $('blind').onchange=controls;
  $('profile').onchange=()=>{clearSession();text('status',meta?.model_available[$('profile').value]?'Model ready.':meta?.errors[$('profile').value]||'Model unavailable.');};
  controls();
  new ResizeObserver(()=>{
    if(window.parent!==window)window.parent.postMessage({type:'qkd-source-height',height:document.body.scrollHeight+40},location.origin);
  }).observe(document.body);
  (async()=>{
    try{
      meta=await api('meta');meta.scenarios.forEach((name,i)=>{const o=document.createElement('option');o.value=i;o.textContent=name;$('scenario').append(o);});
      text('status',meta.model_available[$('profile').value]?'Source model available.':meta.errors[$('profile').value]);controls();
    }catch(error){text('status',error.message);return;}
    try{
      const data=await api('benchmarks');
      text('bench',data.rows.filter(r=>r.regime==='nominal'&&['combined','receiver'].includes(r.detector)&&['minimal','instrumented'].includes(r.variant)).map(r=>
        `${r.variant} / ${r.detector}: ${((r.detection_rate||0)*100).toFixed(1)}% exposure detection; ${r.missed_events} missed; ${r.false_alarm_events} false alarms in ${r.benign_hours.toFixed(2)} benign hours; ${r.false_alarms_per_hour?.toFixed(2)}/h.`).join('\n'));
    }catch(error){text('bench','Benchmarks unavailable: '+error.message);}
  })();
})();
