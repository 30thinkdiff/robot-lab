const assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),path=require('node:path');
(async()=>{
 const {IDS,presetPanels,cleanState,parseStored,pixels,fractions,fitRect}=await import(pathToFileURL(path.resolve(__dirname,'../web/workspace_model.mjs')));
 const checks=[];
 for(const preset of ['review','heatmap','compare']){
  const p=presetPanels(preset);assert.deepEqual(Object.keys(p),IDS);assert(Object.values(p).filter(v=>v.visible).length>=3);
 }checks.push('three meaningful presets cover whitelisted modules');
 for(const raw of [null,'invalid','[]','{"version":9,"enabled":true}', 'x'.repeat(65537)])assert.equal(parseStored(raw).enabled,false);
 const bad=cleanState({version:1,enabled:'yes',preset:'__proto__',panels:{replay:{x:-9,y:Infinity,w:0,h:99,visible:true}},order:['bogus','replay','replay'],named:[{id:'../../evil',name:'x'}]});
 assert.equal(bad.enabled,false);assert.equal(bad.preset,'review');assert.deepEqual(bad.panels.replay,presetPanels().replay);assert.equal(bad.named.length,0);assert.equal(new Set(bad.order).size,IDS.length);
 checks.push('corrupt storage, invalid schema, prototype names, duplicate order and unknown panels safely rejected');
 for(const [w,h]of [[1920,956],[1366,655],[900,700],[320,380]]){
  for(const p of Object.values(presetPanels())){
   const b=pixels(p,w,h,420,240);assert(b.x>=0&&b.y>=0&&b.x+b.w<=w+.001&&b.y+b.h<=h+.001);
   const f=fractions(b,w,h,p.visible);const restored=pixels(f,w,h,420,240);for(const k of ['x','y','w','h'])assert(Math.abs(restored[k]-b[k])<.01);
  }
 }assert.deepEqual(fitRect({x:-99,y:999,w:999,h:1},200,100,420,240),{x:0,y:0,w:200,h:100});
 assert(Object.values(fractions({x:0,y:0,w:1,h:1},0,0)).every(v=>typeof v==='boolean'||Number.isFinite(v)));
 checks.push('minimum sizes and viewport clamping preserve reachable panels on desktop, laptop and narrow screens');
 const named={version:1,enabled:true,locked:true,preset:'user_saved',panels:presetPanels('heatmap'),order:IDS,named:[{id:'user_saved',name:'我的复盘',locked:true,panels:presetPanels('heatmap')}]};
 assert.deepEqual(parseStored(JSON.stringify(named)),named);assert.equal(cleanState({...named,named:Array.from({length:10},(_,i)=>({...named.named[0],id:'user_'+i}))}).named.length,8);
 checks.push('named layout geometry, visibility, lock state and eight-layout limit round-trip');
 console.log(JSON.stringify({status:'passed',checks},null,2));
})().catch(e=>{console.error(e);process.exit(1)});
