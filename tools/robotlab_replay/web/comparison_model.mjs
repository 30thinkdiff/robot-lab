export function visibleByFlags(run,{videoOnly=false,includeForced=false}={}){
 return (!videoOnly||run.video_available===true)&&(includeForced||!run.forced);
}
export function comparisonCandidates(catalog,primary,{scope='same',text='',...flags}={}){
 const query=text.trim().toLocaleLowerCase();
 return catalog.filter(r=>r.id!==primary.id&&visibleByFlags(r,flags)&&
  (scope==='all'||r.condition===primary.condition&&r.seed===primary.seed)&&
  (!query||[r.method,r.model,r.experiment,r.condition,`seed ${r.seed}`,`task_${r.task}`,`init_${r.init}`].join(' ').toLocaleLowerCase().includes(query)));
}
export function outcomeCode(group,methods){
 return methods.map(m=>group.methods[m]?.success===true?'1':group.methods[m]?.success===false?'0':'?').join('');
}
export function selectCaseGroups(groups,methods,{study='',phase='',complete=false,videoOnly=false,allowPartial=false,outcomes={}}={}){
 if(methods.length<2||new Set(methods).size!==methods.length)return [];
 return groups.filter(g=>(!study||g.study===study)&&(!phase||g.phase===phase)&&
  methods.some(m=>g.methods[m])&&
  (allowPartial||methods.every(m=>g.methods[m]))&&
  (!complete||methods.every(m=>g.methods[m]?.replay_complete))&&
  (!videoOnly||methods.every(m=>g.methods[m]?.video))&&
  methods.every(m=>!outcomes[m]||(outcomes[m]==='missing'?!g.methods[m]||typeof g.methods[m].success!=='boolean':
    outcomes[m]==='success'?g.methods[m]?.success===true:g.methods[m]?.success===false))
 ).map(g=>({...g,pattern:outcomeCode(g,methods)}));
}
