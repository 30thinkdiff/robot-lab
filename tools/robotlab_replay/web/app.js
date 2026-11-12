const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=v=>typeof v==='number'&&Number.isFinite(v);
const fmt=(v,n=3)=>number(v)?v.toFixed(n):'未记录';
const pct=v=>number(v)?(v*100).toFixed(1)+'%':'未确定';
const state={catalog:[],a:null,b:null,step:0,playing:false,compare:false,speed:1,limit:65,request:0,pairs:[]};
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data;}
function toast(message){$('toast').textContent=message;$('toast').classList.remove('hidden');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.add('hidden'),4500);}
function fail(message){$('errorBanner').textContent=message;$('errorBanner').classList.remove('hidden');}
function statusName(r){return r.status!=='completed'?'程序未完成':r.success?'任务成功':'任务失败';}
function runLabel(r){return `${r.method} · ${r.model} · task_${r.task} / init_${r.init} · seed ${r.seed}`;}
function options(id,field,label){const values=[...new Set(state.catalog.map(r=>r[field]).filter(v=>v!==null&&v!==undefined))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b)));$(id).innerHTML='<option value="">'+label+'</option>'+values.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');}
function filtered(){const text=$('search').value.trim().toLowerCase();return state.catalog.filter(r=>
  (!$('experimentFilter').value||r.experiment===$('experimentFilter').value)&&
  (!$('modelFilter').value||r.model===$('modelFilter').value)&&(!$('methodFilter').value||r.method===$('methodFilter').value)&&
  (!$('taskFilter').value||String(r.task)===$('taskFilter').value)&&(!$('initFilter').value||String(r.init)===$('initFilter').value)&&
  (!$('seedFilter').value||String(r.seed)===$('seedFilter').value)&&
  (!$('outcomeFilter').value||($('outcomeFilter').value==='error'?r.status!=='completed':String(r.success)===$('outcomeFilter').value))&&
  (!$('videoOnly').checked||r.video_available)&&($('includeForced').checked||!r.forced)&&
  (!text||(runLabel(r)+' '+r.experiment+' '+r.condition).toLowerCase().includes(text)));
}
function renderList(){const rows=filtered();$('filterCount').textContent=`${rows.length} 个回合`;$('loadMore').classList.toggle('hidden',rows.length<=state.limit);
 $('runList').innerHTML=rows.slice(0,state.limit).map(r=>`<button class="run-item ${state.a?.run.id===r.id?'active':''}" data-id="${r.id}" title="${esc(runLabel(r))}"><div class="line1"><span><i class="dot ${r.success?'success':'failure'}"></i>${esc(r.method)} <span class="muted">${esc(r.model)}</span></span>${r.video_available?'▣':''}</div><div class="line2">task_${esc(r.task)} · init_${esc(r.init)} · seed ${esc(r.seed)}</div><div class="line3"><span class="mini-tag">${r.steps??'—'} steps</span><span>${esc(statusName(r))}</span>${r.forced?'<span>接口检查</span>':''}</div></button>`).join('')||'<p class="empty-detail">没有符合筛选条件的回合。</p>';
 $('runList').querySelectorAll('button[data-id]').forEach(b=>b.onclick=()=>selectRun(b.dataset.id));
}
function maxSteps(){return Math.max(1,state.a?.steps.length||0,state.b?.steps.length||0);}
function pause(){state.playing=false;$('play').textContent='▶';$('videoA').pause();$('videoB').pause();}
function setVideo(side,detail){const v=$('video'+side),missing=$('missing'+side);v.pause();v.removeAttribute('src');
 if(detail?.video.available){v.controls=detail.video.alignment!=='recording_verified';v.src=detail.video.url;v.load();missing.classList.add('hidden');v.classList.remove('hidden');v.onloadedmetadata=()=>syncVideo(side,detail,true);}
 else{v.classList.add('hidden');missing.classList.remove('hidden');}
 $('label'+side).textContent=detail?`${detail.run.method} / ${detail.run.model}`:'—';
 $('sync'+side).textContent=(detail?.video.message||'没有对照回合')+(detail?.video.available&&detail.video.alignment!=='recording_verified'?' 可用视频原生控件独立播放；数据联动已禁用。':'');
}
async function selectRun(id){pause();const ticket=++state.request;$('errorBanner').classList.add('hidden');
 try{const [detail,pairs]=await Promise.all([api('/api/run?id='+id),api('/api/pairs?id='+id)]);if(ticket!==state.request)return;
  state.a=detail;state.b=null;state.step=0;state.pairs=pairs;$('note').value='';setVideo('A',detail);renderHeader();renderList();
  $('pairSelect').innerHTML=pairs.map(r=>`<option value="${r.id}">${esc(r.method+' / '+r.model+' · '+statusName(r))}</option>`).join('');
  if(state.compare){const best=pairs.find(r=>r.method==='A'&&r.experiment==='混合 Corrector')||pairs.find(r=>r.success!==detail.run.success)||pairs[0];
    if(best){$('pairSelect').value=best.id;await selectPair(best.id);}else{state.compare=false;toast('没有同任务、初始状态与种子的可用配对。');}}
  applyMode();drawAll();
 }catch(e){fail('打开回合失败：'+e.message);}
}
async function selectPair(id){pause();try{const d=await api('/api/run?id='+id);if(!state.a)return;
 if(d.run.condition!==state.a.run.condition||d.run.seed!==state.a.run.seed)throw Error('配对条件不一致');
 state.b=d;setVideo('B',d);applyMode();drawAll();}catch(e){fail('配对读取失败：'+e.message);}}
function applyMode(){const paired=state.compare&&!!state.b;$('paneB').classList.toggle('hidden',!paired);$('plotB').classList.toggle('hidden',!paired);$('players').classList.toggle('paired',paired);
 $('pairSummary').classList.toggle('hidden',!paired);$('inspectLane').classList.toggle('hidden',!paired);if(!paired)$('inspectLane').value='a';
 $('comparePicker').classList.toggle('hidden',!state.compare);$('pairNotice').classList.toggle('hidden',!paired);$('singleTab').classList.toggle('active',!state.compare);$('compareTab').classList.toggle('active',state.compare);
 if(paired){const a=state.a.metrics,b=state.b.metrics;let sa=a.initial_signature||[a.initial_state_sha256,a.first_input_sha256,a.first_noise_sha256],sb=b.initial_signature||[b.initial_state_sha256,b.first_input_sha256,b.first_noise_sha256];
  const known=sa.length===3&&sb.length===3&&sa.every(Boolean)&&sb.every(Boolean);$('pairNotice').textContent='按相同环境步定位，不代表相同操作阶段；已结束回合显示冻结末帧。'+(known?(JSON.stringify(sa)===JSON.stringify(sb)?' 初始状态 / 输入 / 首噪声SHA一致。':' 注意：初始签名不同，不能作为严格控制变量对照。'):' 原日志缺少完整初始签名，配对仅按任务 / init / seed。');
  const firstEvent=d=>d.events.find(e=>e.kind==='replan')?.step??'未记录';
  const fields=[['任务结果',d=>statusName(d.run)],['执行步数',d=>d.metrics.env_steps],['VLA块调用',d=>d.metrics.chunk_calls],['重规划事件',d=>d.metrics.event_replans],['OGG引导调用',d=>d.metrics.guidance_calls],['VLA累计推理',d=>fmt(d.metrics.policy_s)+' s'],['监测累计耗时',d=>fmt(d.metrics.monitor_s)+' s'],['回合总耗时',d=>fmt(d.metrics.episode_wall_s)+' s'],['监测覆盖',d=>/^H[0-9]+$/.test(d.run.method)?'不适用':pct(d.coverage?.fraction)],['最长空窗',d=>(d.coverage?.max_gap_steps??'未记录')+' steps'],['首次中断',d=>firstEvent(d)],['GPU峰值分配',d=>fmt(d.metrics.peak_torch_allocated_gib)+' GiB']];
  $('pairTable').innerHTML='<table><thead><tr><th>指标</th><th>主选 · '+esc(state.a.run.method+' / '+state.a.run.model)+'</th><th>对照 · '+esc(state.b.run.method+' / '+state.b.run.model)+'</th></tr></thead><tbody>'+fields.map(([name,get])=>`<tr><td>${name}</td><td>${esc(get(state.a)??'未记录')}</td><td>${esc(get(state.b)??'未记录')}</td></tr>`).join('')+'</tbody></table>';}
}
async function mode(compare){if(!state.a)return;pause();state.compare=compare;if(!compare){state.b=null;applyMode();drawAll();return;}
 if(!state.pairs.length){state.compare=false;toast('此回合没有同条件配对。');return;}
 const best=state.pairs.find(r=>r.method==='A'&&r.experiment==='混合 Corrector')||state.pairs.find(r=>r.success!==state.a.run.success)||state.pairs[0];$('pairSelect').value=best.id;await selectPair(best.id);}
function renderHeader(){const d=state.a,r=d.run,m=d.metrics;$('eyebrow').textContent=r.experiment.toUpperCase()+' / LIBERO OBJECT';
 const baseline=/^H[0-9]+$/.test(r.method);$('title').textContent=`Task ${r.task} · ${baseline?'SmolVLA':r.model}`;
 const role=baseline?`纯VLA，${r.model}为归档对照组标签`:{A:'单动作检测 + 单动作OGG',B:'单动作检测 + A10引导',C:'A10检测 + A10引导'}[r.method];
 $('subtitle').textContent=`${r.condition}  /  seed ${r.seed}  /  ${r.method}  /  ${r.forced?'强制接口检查':'真实完整任务回合'}${role?' · '+role:''}`;
 $('outcomeBadge').textContent=statusName(r);$('outcomeBadge').className='outcome '+(r.success?'success':'failure');$('forcedWarning').classList.toggle('hidden',!r.forced);
 const cards=[['环境控制步',m.env_steps,'steps',`回合墙钟 ${fmt(m.episode_wall_s,2)} s`],['VLA动作块',m.chunk_calls,'calls','完整动作块生成'],['纠错 / 重规划',m.event_replans,'events',`OGG引导 ${m.guidance_calls??'未记录'} 次`],['监测覆盖率',baseline?'不适用':pct(d.coverage?.fraction),'',baseline?'纯VLA，未启用Corrector':d.coverage?`最长空窗 ${d.coverage.max_gap_steps} 步`:'缺少分母协议'],['VLA / 监测',`${fmt(m.policy_s,2)} / ${fmt(m.monitor_s,2)}`,'s',`峰值分配显存 ${fmt(m.peak_torch_allocated_gib,2)} GiB`]];
 $('metrics').innerHTML=cards.map(([label,value,unit,sub])=>`<div class="metric"><label>${label}</label><strong ${String(value).length>10?'style="font-size:16px"':''}>${esc(value??'—')}</strong><small>${unit}</small><div class="sub">${esc(sub)}</div></div>`).join('');
 $('syncBadge').textContent=d.video.alignment==='recording_verified'?'录制协议 + PTS已核验':d.video.available?'同步精度受限':'视频缺失 · trace可用';
 $('provenance').textContent=`来源：${d.source.archive} · ${d.source.member}\nTrial SHA256：${d.source.trial_sha256}\n${d.limits.join('；')}\n${d.warnings.join('；')}`;
 $('protocol').textContent=JSON.stringify({source:d.source,video_alignment:d.video.alignment,recording_evidence:d.video.recorder,protocol:d.protocol},null,2);
}
function gaps(detail){const result=[];let start=null;for(const row of detail.steps){const gap=detail.first_due!==null&&row.step>=detail.first_due&&!number(row.score)&&row.monitor_state!=='未记录';
 if(gap&&start===null)start=row.step;if(!gap&&start!==null){result.push([start,row.step-1]);start=null;}}
 if(start!==null)result.push([start,detail.steps.length-1]);return result;}
function drawPlot(id,d,label){const svgWidth=1000,H=208,left=44,right=16,top=22,bottom=40,W=svgWidth-left-right,plotH=H-top-bottom,total=Math.max(1,maxSteps()-1);const X=n=>left+n/total*W;
 const scores=d.steps.flatMap(s=>[s.score,s.executed_score]).filter(number);const thresholds=d.steps.map(s=>s.threshold).filter(v=>number(v)&&v<=2);
 const ymax=Math.max(1,...scores,...thresholds)*1.1;const Y=v=>top+plotH*(1-v/ymax);let svg=`<svg viewBox="0 0 ${svgWidth} ${H}" preserveAspectRatio="none" role="img" aria-label="${esc(label)} LVM时间轴"><rect x="${left}" y="${top}" width="${W}" height="${plotH}" fill="#fcfdfe"/>`;
 if(d.first_due!==null)svg+=`<rect x="${left}" y="${top}" width="${Math.min(W,X(d.first_due)-left)}" height="${plotH}" fill="#f0f3f7"/>`;
 for(const [start,end]of gaps(d))svg+=`<rect x="${X(start)}" y="${top}" width="${Math.max(1,X(end+1)-X(start))}" height="${plotH}" fill="#fbf0df"/>`;
 if(d.steps.length<maxSteps())svg+=`<rect x="${X(d.steps.length)}" y="${top}" width="${Math.max(0,X(maxSteps()-1)-X(d.steps.length))}" height="${plotH}" fill="#edf0f5"/>`;
 for(let i=0;i<5;i++){const val=ymax*i/4,y=Y(val);svg+=`<path d="M${left} ${y} H${svgWidth-right}" stroke="#e9eef4"/><text x="${left-8}" y="${y+3}" text-anchor="end" fill="#9ca8b5" font-size="9">${val.toFixed(2)}</text>`;}
 for(let i=0;i<=5;i++){const step=Math.round(total*i/5);svg+=`<text x="${X(step)}" y="${H-9}" text-anchor="middle" fill="#9ca8b5" font-size="9">${step}</text>`;}
 for(const [key,color]of[['score','#247cbd'],['threshold','#d69232'],['executed_score','#189a90']]){let path='',connected=false;
  for(const row of d.steps){const val=row[key];if(!number(val)||val>ymax){connected=false;continue;}path+=(connected?' L':' M')+X(row.step).toFixed(2)+','+Y(val).toFixed(2);connected=true;}
  svg+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="1.8" ${key==='threshold'?'stroke-dasharray="4 3"':''}/>`;}
 const clipped=d.steps.filter(r=>number(r.threshold)&&r.threshold>ymax);if(clipped.length)svg+=`<text x="${svgWidth-right-3}" y="12" text-anchor="end" fill="#bf8e4c" font-size="8">初始Threshold ${Math.max(...clipped.map(r=>r.threshold)).toFixed(1)} 超出图范围，未改写数值</text>`;
 const seen=new Set();for(const e of d.events){if(!['replan','ogg','target'].includes(e.kind))continue;const key=e.step+e.kind;if(seen.has(key))continue;seen.add(key);const color=e.kind==='ogg'?'#8f76c5':e.kind==='target'?'#189a90':'#d69232';const y=H-29+(e.kind==='ogg'?6:0);svg+=`<circle cx="${X(e.step)}" cy="${y}" r="3" fill="${color}"><title>step ${e.step} · ${esc(e.label)}</title></circle>`;}
 for(const row of d.steps)if(row.block_start_derived)svg+=`<path d="M${X(row.step)} ${top+plotH+1} v4" stroke="#89a4b8"/>`;
 svg+=`<text x="${left+6}" y="13" font-size="9" fill="#7d8c9c">${esc(label)}</text><path d="M${X(state.step)} ${top-3} V${H-21}" stroke="#304f6b" stroke-width="1"/><circle cx="${X(state.step)}" cy="${top-3}" r="2.8" fill="#304f6b"/></svg>`;
 $(id).innerHTML=svg;$(id).onclick=e=>{$('inspectLane').value=id==='plotB'?'b':'a';const rect=$(id).querySelector('svg').getBoundingClientRect();setStep(Math.round(((e.clientX-rect.left)/rect.width*svgWidth-left)/W*total),true);};
}
function inspect(){const d=$('inspectLane').value==='b'&&state.b?state.b:state.a,s=d.steps[state.step];$('inspectStep').textContent=String(state.step).padStart(3,'0');$('rawStep').textContent=JSON.stringify(s?.raw??[],null,2);
 $('stepContext').innerHTML='<table><thead><tr><th>步</th><th>Score</th><th>阈值</th><th>事件</th></tr></thead><tbody>'+d.steps.slice(Math.max(0,state.step-5),state.step+6).map(r=>`<tr ${r.step===state.step?'class="current-context"':''}><td>${r.step}</td><td>${number(r.score)?fmt(r.score,3):'—'}</td><td>${number(r.threshold)?fmt(r.threshold,3):'—'}</td><td>${r.events.length||'—'}</td></tr>`).join('')+'</tbody></table>';
 if(!s){$('readout').innerHTML='<div class="inspect-status missing">当前回合已结束，不沿用末次评分</div>';$('actions').innerHTML='<div class="empty-detail">超出此回合记录范围</div>';$('stepEvents').innerHTML='<div class="empty-detail">没有该步数据</div>';return;}
 const row=(label,value,cls='')=>`<div class="read-row"><span>${label}</span><strong class="${cls}">${esc(value)}</strong></div>`;
 $('readout').innerHTML=`<div class="inspect-status ${s.monitor_state==='无到期评分'?'gap':['未记录','未启用Corrector'].includes(s.monitor_state)?'missing':''}">${esc(s.monitor_state)}</div>`+
  row('LVM Score',fmt(s.score,4),'blue-text')+row('Threshold',fmt(s.threshold,4))+row('已执行条件Score',fmt(s.executed_score,4))+
  row('动作块号（推导）',s.block_derived??'未记录')+row('剩余队列（推导）',s.queue_remaining_derived??'未记录')+
  (number(s.expected_norm)?row('预测变化范数',fmt(s.expected_norm,2)):'')+(number(s.observed_norm)?row('实测变化范数',fmt(s.observed_norm,2)):'')+
  (number(s.derived_mse)?row('残差MSE（推导）',fmt(s.derived_mse,3)):'');
 const guides=s.events.filter(e=>e.kind==='ogg'),targets=s.events.filter(e=>e.kind==='target'),interrupt=s.events.find(e=>e.kind==='replan');let content='';
 if(interrupt)content+=`<div class="event-detail"><strong>队列中断 / 立即重规划</strong><p>丢弃动作 ${esc(interrupt.data.discarded_actions??'未记录')} 条；本步记录的Score ${fmt(s.score,4)}。</p></div>`;
 if(guides.length){const loss=guides.map(e=>e.data.ogg_loss??e.data.guidance_loss).filter(number),grad=guides.map(e=>e.data.gradient_l2).filter(number),update=guides.map(e=>e.data.clean_action_update_l2).filter(number);
  content+=`<div class="event-detail"><strong>OGG · ${guides.length}次已记录引导</strong><p>Loss ${loss.length?loss.map(v=>fmt(v,4)).join(' → '):'当前实验未记录'}<br>梯度范数 ${grad.length?grad.map(v=>fmt(v,5)).join(' / '):'当前实验未记录'}<br>动作更新L2 ${update.length?update.map(v=>fmt(v,5)).join(' / '):'当前实验未记录'}</p><p>更新量为候选归一化动作变化，不是机器人测量位移。</p></div>`;}
 if(targets.length){const t=targets[0].data;content+=`<div class="event-detail"><strong>纠偏目标 · ${esc(t.target_shape?.join(' × ')||'shape未记录')}</strong><p>origin step ${esc(t.prediction_origin_step??'未记录')}<br>目标范数 ${fmt(t.target_norm,2)}<br>原始Tensor未导入，不反序列化.pt。</p></div>`;}
 $('stepEvents').innerHTML=content||'<div class="empty-detail">本步未记录中断或OGG调用。<br>完整预测Tensor / 候选动作Tensor：当前实验未记录或未导入。</div>';
 const action=s.submitted_action||s.controller_command?.raw_controller_action;
 if(Array.isArray(action)){const scale=Math.max(.001,...action.map(Math.abs));$('actions').innerHTML=`<p class="empty-detail">uₜ：提交给环境的控制命令<br>非机器人测量状态；条形按当前幅值缩放。</p><div class="action-vector">`+action.map((v,i)=>{const w=Math.abs(v)/scale*46;return `<div class="action-component"><span>a${i}</span><div class="action-track"><i style="left:${v>=0?50:50-w}%;width:${w}%"></i></div><code>${fmt(v,3)}</code></div>`;}).join('')+'</div>';}
 else $('actions').innerHTML='<div class="empty-detail">当前实验未记录 / 未导入逐步动作或机器人状态。不会从视频估造动作值。</div>';
}
function renderEvents(){if(!state.a)return;const filter=$('eventFilter').value;const seen=new Map();const events=[...state.a.events];
 for(const [a,b]of gaps(state.a))events.push({step:a,kind:'gap',label:`监测空窗 · step ${a}—${b}`,data:{derived:true,length:b-a+1}});
 for(const e of events){if(filter==='important'&&!['replan','ogg','target'].includes(e.kind))continue;if(!['all','important'].includes(filter)&&e.kind!==filter)continue;const key=e.step+':'+e.kind;if(seen.has(key)){seen.get(key).count++;continue;}seen.set(key,{...e,count:1});}
 const grouped=[...seen.values()].sort((a,b)=>a.step-b.step);$('eventCount').textContent=`${new Set(grouped.map(e=>e.step)).size} 步 · ${grouped.length} 组`;
 $('events').innerHTML=grouped.map(e=>`<button class="event-row ${e.step===state.step?'selected':''}" data-step="${e.step}"><span class="event-step">${String(e.step).padStart(3,'0')}</span><span><i class="event-kind ${e.kind}"></i>${esc(e.label)}${e.count>1?' × '+e.count:''}<small>${e.kind==='gap'?'由缺失评分推导，非原始异常标签':state.a.control_fps?(e.step/state.a.control_fps).toFixed(2)+' s 仿真控制时间':'没有记录控制频率'}</small></span><span class="muted">↗</span></button>`).join('')||'<p class="empty-detail" style="padding:0 18px 16px">该筛选下没有记录的事件。</p>';
 $('events').querySelectorAll('button[data-step]').forEach(b=>b.onclick=()=>setStep(Number(b.dataset.step),true));
}
function syncVideo(side,d,force=false){if(!d)return;const v=$('video'+side),valid=d.video.alignment==='recording_verified',end=state.step>=d.steps.length;
 $('step'+side).textContent=end?`已结束 · ${d.steps.length}步`:`step ${state.step}`;$('ended'+side).classList.toggle('hidden',!end);$('ended'+side).textContent=`已结束，冻结末帧 ${Math.max(0,d.steps.length-1)}`;
 if(!valid||!v.src)return;const pts=d.video.pts,index=Math.min(state.step,pts.length-1),time=pts[index]+.001;
 if(force||Math.abs(v.currentTime-time)>.085||(end&&Math.abs(v.currentTime-time)>.004)){v.currentTime=Math.min(time,Math.max(0,(v.duration||Infinity)-.002));}
 if(end)v.pause();
}
function stepFromTime(d,time){const pts=d.video.pts;let lo=0,hi=pts.length-1;while(lo<=hi){const mid=(lo+hi)>>1;if(pts[mid]<=time+.0001)lo=mid+1;else hi=mid-1;}return Math.max(0,hi);}
function setStep(value,manual=false){if(!state.a)return;if(manual)pause();state.step=Math.max(0,Math.min(maxSteps()-1,Math.round(value)||0));drawAll(true);}
function drawAll(forceSeek=false){if(!state.a)return;const n=maxSteps();state.step=Math.min(state.step,n-1);$('seek').max=n-1;$('seek').value=state.step;$('stepInput').max=n-1;$('stepInput').value=state.step;
 $('clockStep').textContent=String(state.step).padStart(3,'0');$('clockTotal').textContent='/ '+String(n).padStart(3,'0')+' 步';$('rangeEnd').textContent='step '+(n-1);
 const fps=state.a.control_fps,pts=state.a.video.pts;const videoTime=state.a.video.alignment==='recording_verified'?pts?.[state.step]:null;$('clockTime').textContent=`控制 ${fps?(state.step/fps).toFixed(2)+' s':'频率未记录'} · 视频PTS ${number(videoTime)?videoTime.toFixed(3)+' s':'无经核验映射'}`;
 syncVideo('A',state.a,forceSeek);if(state.b)syncVideo('B',state.b,forceSeek);
 drawPlot('plot',state.a,'A · '+state.a.run.method);if(state.b)drawPlot('plotB',state.b,'B · '+state.b.run.method);
 const a=state.a.coverage,b=state.b?.coverage;$('coverageLabel').textContent=(/^H[0-9]+$/.test(state.a.run.method)?'A：纯VLA，监测覆盖不适用':`A覆盖 ${pct(a?.fraction)} · 最长空窗 ${a?.max_gap_steps??'—'}步`)+(state.b?(/^H[0-9]+$/.test(state.b.run.method)?' / B：纯VLA，不适用':` / B覆盖 ${pct(b?.fraction)} · 空窗 ${b?.max_gap_steps??'—'}步`):'');
 inspect();renderEvents();
}
function play(){if(!state.a)return;if(state.playing){pause();return;}if(state.step>=maxSteps()-1)setStep(0,true);state.playing=true;$('play').textContent='Ⅱ';
 const details=[['A',state.a],['B',state.b]].filter(([,d])=>d&&d.video.alignment==='recording_verified');
 const master=details.sort((a,b)=>b[1].steps.length-a[1].steps.length)[0];
 for(const [side,d]of details){const v=$('video'+side);v.playbackRate=state.speed;if(state.step<d.steps.length)v.play().catch(e=>{pause();toast('浏览器尚未准备好视频，请稍后再播放。');});}
 const start=performance.now(),origin=state.step;
 function tick(now){if(!state.playing)return;let step;
  if(master){const v=$('video'+master[0]);step=stepFromTime(master[1],v.currentTime);if(v.ended)step=maxSteps()-1;}
  else step=origin+Math.floor((now-start)/1000*(state.a.control_fps||1)*state.speed);
  if(step!==state.step){state.step=Math.min(maxSteps()-1,step);drawAll();}if(state.step>=maxSteps()-1){pause();return;}requestAnimationFrame(tick);}
 requestAnimationFrame(tick);
}
$('singleTab').onclick=()=>mode(false);$('compareTab').onclick=()=>mode(true);$('pairSelect').onchange=()=>selectPair($('pairSelect').value);
$('focusBtn').onclick=()=>{const focus=document.body.classList.toggle('focus-mode');$('focusBtn').textContent=focus?'显示概览':'专注回放';$('focusBtn').setAttribute('aria-pressed',String(focus));};
$('inspectLane').onchange=inspect;
$('play').onclick=play;$('prev').onclick=()=>setStep(state.step-1,true);$('next').onclick=()=>setStep(state.step+1,true);$('seek').oninput=e=>setStep(Number(e.target.value),true);$('stepInput').onchange=e=>setStep(Number(e.target.value),true);
$('speed').onchange=e=>{const was=state.playing;pause();state.speed=Number(e.target.value);if(was)play();};$('eventFilter').onchange=renderEvents;
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||$('exportDialog').open)return;if(e.code==='Space'){e.preventDefault();play();}if(e.code==='ArrowLeft'){e.preventDefault();setStep(state.step-1,true);}if(e.code==='ArrowRight'){e.preventDefault();setStep(state.step+1,true);}});
for(const id of ['experimentFilter','modelFilter','methodFilter','taskFilter','initFilter','seedFilter','outcomeFilter','videoOnly','includeForced'])$(id).onchange=()=>{state.limit=65;renderList();};
$('search').oninput=()=>{state.limit=65;renderList();};$('clearFilters').onclick=()=>{for(const id of ['experimentFilter','modelFilter','methodFilter','taskFilter','initFilter','seedFilter','outcomeFilter','search'])$(id).value='';$('videoOnly').checked=$('includeForced').checked=false;renderList();};
$('loadMore').onclick=()=>{state.limit+=100;renderList();};$('closeExport').onclick=()=>$('exportDialog').close();
$('exportBtn').onclick=async()=>{if(!state.a)return;pause();$('exportBtn').disabled=true;try{const ids=[state.a.run.id,...(state.b?[state.b.run.id]:[])];const result=await api('/api/export',{ids,step:state.step,note:$('note').value});$('exportDir').textContent=result.directory;$('exportFiles').innerHTML=result.files.map(name=>`<a download="${esc(name)}" href="/api/export-file/${result.export}/${esc(name)}">${esc(name)}</a>`).join('');$('exportDialog').showModal();}catch(e){toast('导出失败：'+e.message);}finally{$('exportBtn').disabled=false;}};
window.addEventListener('pagehide',pause);
async function boot(){try{const catalog=await api('/api/catalog');state.catalog=catalog.runs;$('catalogCount').textContent=state.catalog.length+' 条';
 for(const [id,field,label]of[['experimentFilter','experiment','全部实验'],['modelFilter','model','全部模型'],['methodFilter','method','全部方法'],['taskFilter','task','全部任务'],['initFilter','init','全部init'],['seedFilter','seed','全部seed']])options(id,field,label);
 renderList();const first=state.catalog.find(r=>r.video_available&&r.method==='B'&&!r.forced&&r.task===1&&r.init===0)||state.catalog.find(r=>r.video_available&&!r.success&&!r.forced)||state.catalog.find(r=>!r.success&&!r.forced)||state.catalog[0];
 if(first)await selectRun(first.id);else fail('没有发现可读取的实验。使用 --source 指定只读评测目录或结果ZIP。');
 if(catalog.warnings.length)toast('部分来源无法读取；其他回合仍可用。详见服务日志/来源说明。');
}catch(e){fail('索引加载失败：'+e.message);}}
boot();
