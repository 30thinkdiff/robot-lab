import {IDS,PRESETS,presetPanels,cleanPanels,parseStored,pixels,fitRect,fractions} from './workspace_model.mjs';

const KEY='robotlab-replay.workspace.v1';
const $=id=>document.getElementById(id);
const descriptors={
 library:{name:'实验资料库',selectors:['.library'],min:[220,240]},
 overview:{name:'回合概览与配置',selectors:['#errorBanner','.heading','#forcedWarning','.config-guide','#metrics'],min:[360,200]},
 replay:{name:'操作回放与时间轴',selectors:['.playback-card','#anchorTools'],min:[420,300]},
 timeline:{name:'LVM与纠偏曲线',selectors:['.timeline-card'],min:[340,200]},
 inspector:{name:'当前步诊断',selectors:['.inspector'],min:[260,230]},
 events:{name:'关键事件',selectors:['.event-card'],min:[300,200]},
 evidence:{name:'证据摘要与动作热图',selectors:['.evidence-section'],min:[380,230]},
 comparison:{name:'配对指标',selectors:['#pairSummary'],min:[340,200]},
 protocol:{name:'来源与冻结协议',selectors:['.technical'],min:[340,200]}
};
const copy=value=>JSON.parse(JSON.stringify(value));

export function createWorkspace({pause,notify}){
 let state;try{state=parseStored(localStorage.getItem(KEY));}catch{state=parseStored(null);}
 const canvas=$('workspaceCanvas'),toolbar=$('workspaceToolbar'),dialog=$('workspaceDialog'),records=new Map();
 let maximized=null,drag=null,saveFailed=false;
 const button=(text,label,handler)=>{const b=document.createElement('button');b.type='button';b.textContent=text;b.title=label;b.setAttribute('aria-label',label);b.onclick=handler;return b;};

 function save(){try{localStorage.setItem(KEY,JSON.stringify(state));$('workspaceStatus').textContent='布局已自动保存';}
  catch{ $('workspaceStatus').textContent='浏览器存储不可用，本次布局仍有效';if(!saveFailed){notify('浏览器未允许保存布局，关闭页面后可能丢失；实验数据不受影响。');saveFailed=true;}}}
 function bounds(){const top=document.querySelector('.topbar').getBoundingClientRect().bottom+toolbar.getBoundingClientRect().height;
  canvas.style.top=Math.round(top)+'px';return {w:Math.max(1,canvas.clientWidth),h:Math.max(1,canvas.clientHeight)};}
 function render(){if(!state.enabled)return;const {w,h}=bounds();
  for(const [id,r]of records){const p=state.panels[id],box=maximized===id?{x:0,y:0,w,h}:pixels(p,w,h,...descriptors[id].min);
   r.panel.hidden=!p.visible||(maximized!==null&&maximized!==id);Object.assign(r.panel.style,{left:box.x+'px',top:box.y+'px',width:box.w+'px',height:box.h+'px',zIndex:state.order.indexOf(id)+1});
   r.panel.classList.toggle('is-maximized',maximized===id);r.panel.classList.toggle('is-locked',state.locked);
   r.maximize.textContent=maximized===id?'↙':'↗';r.maximize.setAttribute('aria-label',maximized===id?'还原'+descriptors[id].name:'最大化'+descriptors[id].name);
  }
  $('workspaceEmpty').hidden=IDS.some(id=>state.panels[id].visible);$('workspaceLock').checked=state.locked;
  $('workspaceToggle').textContent='返回页面布局';$('workspaceToggle').setAttribute('aria-pressed','true');
 }
 function raise(id){state.order=state.order.filter(x=>x!==id).concat(id);if(state.enabled)render();}
 function reveal(id){if(id==='evidence')$('evidenceDetails').open=true;if(id==='protocol')document.querySelector('.technical details').open=true;}
 function maximize(id){if(!state.enabled)return;pause();maximized=maximized===id?null:id;state.panels[id].visible=true;raise(id);render();save();}
 function moveNode(parent,node,before=null){if(node===before)return;if(typeof parent.moveBefore==='function'&&node.isConnected&&parent.isConnected)parent.moveBefore(node,before);else parent.insertBefore(node,before);}
 function show(id,visible){state.panels[id].visible=visible;if(visible||maximized===id)maximized=null;if(visible){raise(id);reveal(id);}render();syncDialog();save();}
 function enable(enabled){pause();state.enabled=enabled;maximized=null;
  if(enabled){document.body.classList.remove('focus-mode');$('focusBtn').textContent='专注回放';$('focusBtn').setAttribute('aria-pressed','false');
   canvas.hidden=false;toolbar.hidden=false;document.body.classList.add('free-workspace');
   for(const r of records.values())for(const placement of r.placements)moveNode(r.body,placement.node,r.empty);
   for(const id of IDS)if(state.panels[id].visible)reveal(id);render();
  }else{for(const r of records.values())for(const p of r.placements)moveNode(p.marker.parentNode,p.node,p.marker.nextSibling);
   document.body.classList.remove('free-workspace');canvas.hidden=true;toolbar.hidden=true;$('workspaceToggle').textContent='自由布局';$('workspaceToggle').setAttribute('aria-pressed','false');}
  syncDialog();save();
 }
 function selectedPreset(){return state.named.find(n=>n.id===state.preset);}
 function applyPreset(id){pause();const named=state.named.find(n=>n.id===id);state.preset=named?id:Object.hasOwn(PRESETS,id)?id:'review';
  state.panels=copy(named?named.panels:presetPanels(state.preset));state.locked=named?.locked??false;maximized=null;
  if(!state.enabled)enable(true);else{for(const id of IDS)if(state.panels[id].visible)reveal(id);render();save();}syncDialog();
 }
 function syncDialog(){const select=$('workspacePreset');select.replaceChildren();
  for(const [id,name]of [...Object.entries(PRESETS),...state.named.map(n=>[n.id,'我的布局 · '+n.name])]){const option=document.createElement('option');option.value=id;option.textContent=name;select.append(option);}select.value=state.preset;
  $('workspacePanelList').replaceChildren();for(const id of IDS){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=state.panels[id].visible;input.dataset.panel=id;input.onchange=()=>show(id,input.checked);label.append(input,document.createTextNode(descriptors[id].name));$('workspacePanelList').append(label);}
  $('workspaceDelete').disabled=!selectedPreset();$('workspaceLock').checked=state.locked;
 }
 function setRect(id,box){const {w,h}=bounds();state.panels[id]=fractions(fitRect(box,w,h,...descriptors[id].min),w,h,true);render();}
 function start(event,id,resizing=false){if(event.button!==0||state.locked||maximized)return;event.preventDefault();raise(id);
  const {w,h}=bounds(),rect=pixels(state.panels[id],w,h,...descriptors[id].min);drag={id,resizing,rect,x:event.clientX,y:event.clientY,pointer:event.pointerId,target:event.currentTarget};drag.target.setPointerCapture(event.pointerId);canvas.classList.add('is-dragging');}
 function update(event){if(!drag||drag.pointer!==event.pointerId)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y,r=drag.rect;
  setRect(drag.id,drag.resizing?{...r,w:r.w+dx,h:r.h+dy}:{...r,x:r.x+dx,y:r.y+dy});}
 function finish(event){if(!drag||drag.pointer!==event.pointerId)return;drag=null;canvas.classList.remove('is-dragging');save();}

 for(const id of IDS){const d=descriptors[id],panel=document.createElement('section');panel.className='workspace-panel';panel.dataset.panel=id;panel.setAttribute('aria-label',d.name+'面板');
  const titlebar=document.createElement('div');titlebar.className='workspace-titlebar workspace-ui';
  const handle=button('⠿ '+d.name,'拖动'+d.name+'；方向键移动，Shift加方向键缩放',()=>{});handle.className='workspace-handle';handle.onpointerdown=e=>start(e,id);
  handle.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();e.stopPropagation();if(state.locked||maximized)return;
   raise(id);const {w,h}=bounds(),r=pixels(state.panels[id],w,h,...d.min),dx=e.key==='ArrowLeft'?-12:e.key==='ArrowRight'?12:0,dy=e.key==='ArrowUp'?-12:e.key==='ArrowDown'?12:0;setRect(id,e.shiftKey?{...r,w:r.w+dx,h:r.h+dy}:{...r,x:r.x+dx,y:r.y+dy});save();};
  handle.ondblclick=()=>maximize(id);
  const controls=document.createElement('div');controls.className='workspace-panel-tools';
  const max=button('↗','最大化'+d.name,()=>maximize(id)),close=button('×','隐藏'+d.name,()=>show(id,false));controls.append(max,close);titlebar.append(handle,controls);
  const body=document.createElement('div');body.className='workspace-body';const empty=document.createElement('p');empty.className='workspace-empty';empty.textContent='切换到配对对比并选择对照后，这里显示真实配对指标。';body.append(empty);
  const placements=d.selectors.map(selector=>{const node=document.querySelector(selector);if(!node)throw Error('布局模块不存在：'+selector);const marker=document.createComment('replay-layout:'+id);node.before(marker);return {node,marker};});
  const checkEmpty=()=>{empty.hidden=placements.some(p=>!p.node.classList.contains('hidden'));};const observer=new MutationObserver(checkEmpty);placements.forEach(p=>observer.observe(p.node,{attributes:true,attributeFilter:['class']}));checkEmpty();
  const resize=button('◢','缩放'+d.name+'；可用标题栏Shift加方向键',()=>{});resize.className='workspace-resize workspace-ui';resize.tabIndex=-1;resize.onpointerdown=e=>start(e,id,true);
  panel.append(titlebar,body,resize);panel.onpointerdown=()=>{if(!drag)raise(id);};panel.onfocusin=()=>raise(id);canvas.append(panel);records.set(id,{panel,body,empty,placements,maximize:max});
 }
 canvas.addEventListener('pointermove',update);canvas.addEventListener('pointerup',finish);canvas.addEventListener('pointercancel',finish);canvas.addEventListener('lostpointercapture',finish);
 $('workspaceToggle').onclick=()=>enable(!state.enabled);$('workspacePreset').onchange=e=>applyPreset(e.target.value);
 $('workspaceManage').onclick=()=>{syncDialog();dialog.showModal();};$('workspaceEmptyOpen').onclick=()=>$('workspaceManage').click();
 $('workspaceDialogClose').onclick=()=>dialog.close();$('workspaceReset').onclick=()=>applyPreset(state.preset);
 $('workspaceLock').onchange=e=>{state.locked=e.target.checked;render();save();};
 $('workspaceShowAll').onclick=()=>{for(const id of IDS){state.panels[id].visible=true;reveal(id);}maximized=null;render();syncDialog();save();};
 $('workspaceSave').onclick=()=>{const name=$('workspaceName').value.trim();if(!name||name.length>32){notify('布局名称需要1～32个字。');return;}
  const existing=state.named.find(n=>n.name===name);if(!existing&&state.named.length>=8){notify('最多保存8套布局，请先删除不需要的一套。');return;}
  const item={id:existing?.id||'user_'+Date.now().toString(36)+'_'+Math.random().toString(36).slice(2,7),name,panels:cleanPanels(state.panels),locked:state.locked};state.named=state.named.filter(n=>n.id!==item.id).concat(item);state.preset=item.id;syncDialog();save();notify('已保存布局：'+name);};
 $('workspaceDelete').onclick=()=>{if(!selectedPreset())return;state.named=state.named.filter(n=>n.id!==state.preset);state.preset='review';syncDialog();save();notify('已删除命名布局；当前面板位置保留。');};
 window.addEventListener('resize',()=>{if(state.enabled)requestAnimationFrame(render);});
 const resizeObserver=new ResizeObserver(()=>{if(state.enabled)requestAnimationFrame(render);});resizeObserver.observe(document.querySelector('.topbar'));resizeObserver.observe(toolbar);
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&state.enabled&&maximized&&!document.querySelector('dialog[open]')){maximized=null;render();}});
 syncDialog();if(state.enabled)enable(true);
 return {isEnabled:()=>state.enabled,toggleMaximize:id=>maximize(id)};
}
