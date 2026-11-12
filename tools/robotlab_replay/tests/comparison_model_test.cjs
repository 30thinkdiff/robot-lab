const assert=require('node:assert/strict');
(async()=>{
 const {comparisonCandidates,outcomeCode,selectCaseGroups}=await import('../web/comparison_model.mjs');
 const a={id:'a',condition:'task1/init0',seed:1},catalog=[a,
  {id:'b',condition:a.condition,seed:1,video_available:true},
  {id:'c',condition:a.condition,seed:1,forced:true,video_available:true},
  {id:'d',condition:a.condition,seed:1,video_available:false},
  {id:'e',condition:'task9/init3',task:9,init:3,seed:2,method:'ogg',model:'A10',video_available:true}];
 assert.deepEqual(comparisonCandidates(catalog,a,{videoOnly:true}).map(r=>r.id),['b']);
 assert.deepEqual(comparisonCandidates(catalog,a,{videoOnly:true,includeForced:true}).map(r=>r.id),['b','c']);
 assert.deepEqual(comparisonCandidates(catalog,a,{scope:'all',videoOnly:true,text:'task9/init3'}).map(r=>r.id),['e']);
 const group={study:'x',phase:'p',methods:{a:{success:true,video:true,replay_complete:true},b:{success:false,video:true,replay_complete:true}}};
 assert.equal(outcomeCode(group,['a','b','missing']),'10?');
 assert.equal(selectCaseGroups([group],['a','b'],{outcomes:{a:'success',b:'failure'}}).length,1);
 assert.equal(selectCaseGroups([group],['a','missing'],{}).length,0);
 assert.equal(selectCaseGroups([group],['a','missing'],{allowPartial:true,outcomes:{missing:'missing'}}).length,1);
 assert.equal(selectCaseGroups([group],['a','missing'],{allowPartial:true,videoOnly:true}).length,0);
 assert.equal(selectCaseGroups([group],['a','missing'],{allowPartial:true,outcomes:{missing:'failure'}}).length,0);
 assert.equal(selectCaseGroups([group],['a','b'],{study:'other'}).length,0);
 assert.equal(selectCaseGroups([group],['a'],{}).length,0);
 assert.equal(selectCaseGroups([group],['x','y'],{allowPartial:true}).length,0);
 console.log('Passed: shared video/forced filters, unrestricted search, missing ≠ failure, method outcomes and cohort isolation.');
})().catch(e=>{console.error(e);process.exit(1)});
