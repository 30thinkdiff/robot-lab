// Pure layout data and bounds: no experiment data, DOM or dependencies.
export const IDS=['library','overview','replay','timeline','inspector','events','evidence','comparison','protocol'];
export const PRESETS={review:'视频 + 曲线 + 诊断',heatmap:'视频 + 动作热图',compare:'并排视频 + 指标对照'};
const clone=value=>JSON.parse(JSON.stringify(value));
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

export function presetPanels(name='review'){
 const panels=Object.fromEntries(IDS.map((id,i)=>[id,{x:.05+(i%3)*.07,y:.05+(i%3)*.06,w:.48,h:.60,visible:false}]));
 const show=(id,x,y,w,h)=>panels[id]={x,y,w,h,visible:true};
 if(name==='heatmap'){show('replay',0,0,.55,.62);show('timeline',0,.62,.55,.38);show('evidence',.55,0,.45,1);}
 else if(name==='compare'){show('replay',0,0,.7,.6);show('timeline',0,.6,.7,.4);show('inspector',.7,0,.3,.6);show('comparison',.7,.6,.3,.4);}
 else{show('replay',0,0,.55,.58);show('timeline',0,.58,.55,.42);show('inspector',.55,0,.25,1);show('library',.8,0,.2,1);}
 return panels;
}

export function cleanPanels(raw,fallback=presetPanels()){
 return Object.fromEntries(IDS.map(id=>{
  const p=raw?.[id];
  const valid=p&&['x','y','w','h'].every(k=>finite(p[k]))&&p.x>=0&&p.y>=0&&p.x<1&&p.y<1&&p.w>0&&p.h>0&&p.w<=1&&p.h<=1&&typeof p.visible==='boolean';
  return [id,valid?{x:p.x,y:p.y,w:p.w,h:p.h,visible:p.visible}:clone(fallback[id])];
 }));
}

export function cleanState(raw){
 const defaults={version:1,enabled:false,locked:false,preset:'review',panels:presetPanels(),order:[...IDS],named:[]};
 if(!raw||raw.version!==1)return defaults;
 const named=[];
 for(const item of Array.isArray(raw.named)?raw.named.slice(0,8):[]){
  if(!item||typeof item.id!=='string'||!/^user_[a-z0-9_]{1,50}$/.test(item.id)||named.some(n=>n.id===item.id)||typeof item.name!=='string'||!item.name.trim()||item.name.length>32)continue;
  named.push({id:item.id,name:item.name,panels:cleanPanels(item.panels),locked:item.locked===true});
 }
 const order=[...new Set((Array.isArray(raw.order)?raw.order:[]).filter(id=>IDS.includes(id)))];
 return {version:1,enabled:raw.enabled===true,locked:raw.locked===true,
  preset:Object.hasOwn(PRESETS,raw.preset)||named.some(n=>n.id===raw.preset)?raw.preset:'review',
  panels:cleanPanels(raw.panels),order:[...order,...IDS.filter(id=>!order.includes(id))],named};
}

export function fitRect(rect,width,height,minWidth=220,minHeight=160){
 width=Math.max(1,width);height=Math.max(1,height);
 const w=clamp(rect.w,Math.min(width,minWidth),width),h=clamp(rect.h,Math.min(height,minHeight),height);
 return {x:clamp(rect.x,0,width-w),y:clamp(rect.y,0,height-h),w,h};
}

export function pixels(panel,width,height,minWidth,minHeight){
 return fitRect({x:panel.x*width,y:panel.y*height,w:panel.w*width,h:panel.h*height},width,height,minWidth,minHeight);
}

export function fractions(rect,width,height,visible=true){
 width=Math.max(1,width);height=Math.max(1,height);
 const p=fitRect(rect,width,height,1,1),round=v=>Math.round(v*1e6)/1e6;
 return {x:round(p.x/width),y:round(p.y/height),w:Math.max(.000001,round(p.w/width)),h:Math.max(.000001,round(p.h/height)),visible};
}

export function parseStored(text){
 try{return typeof text==='string'&&text.length<=65536?cleanState(JSON.parse(text)):cleanState(null);}
 catch{return cleanState(null);}
}
