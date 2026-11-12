import {selectCaseGroups} from './comparison_model.mjs';
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const classicPatterns=['001','000','011','111','010','101','100','110'];
const studyLabel=s=>({legacy_five_models:'旧正式五模型筛选库',hybrid_pilot:'混合引导自然Pilot',consistency_closed_loop:'规划 / 已执行闭环Pilot',horizon_pilot:'SmolVLA Horizon Pilot'})[s]||(s.startsWith('replay_casebank_')?'完整补测 · '+s:s);

export function setupCasebank({api,openPair,notify,getFilters=()=>({})}){
 let bank=null,current=[],selected=['H50','H20','mlp_ogg'],outcomes={};
 const presets={architectures:['mlp_ogg','transformer-h1_ogg','transformer-h3_ogg','transformer-a10-h1_ogg','transformer-a10-h3_ogg'],
  lvm_ogg:['transformer-h3_lvm','transformer-h3_ogg'],hybrid:['transformer-h1_ogg','hybrid_t1_a10_ogg'],consistency:['planned_lvm','executed_lvm'],horizon:['H50','H20','H10','H5','H2','H1']};
 const label=m=>bank.methods.find(r=>r.id===m)?.label||m;
 const patternLabel=p=>selected.map((m,i)=>`${label(m)}${p[i]==='1'?'成功':p[i]==='0'?'失败':'未记录'}`).join(' · ');
 function configure(methods){selected=[...new Set(methods)];outcomes={};$('casebankPattern').innerHTML='<option value="">所有成败组合</option>';
  $('casebankMethodDetails').open=false;
  $('casebankMethods').querySelectorAll('input').forEach(e=>{e.checked=selected.includes(e.value);});
  $('casebankMethodCount').textContent=`已选${selected.length}种`;
  $('casebankOutcomes').innerHTML=selected.map(m=>`<label>${esc(label(m))}<select data-outcome="${esc(m)}" aria-label="${esc(label(m))}结局"><option value="">任何结局</option><option value="success">成功</option><option value="failure">失败</option><option value="missing">未记录</option></select></label>`).join('');
  $('casebankOutcomes').querySelectorAll('select').forEach(e=>e.onchange=()=>{outcomes[e.dataset.outcome]=e.value;if(e.value==='missing')$('casebankAllowPartial').checked=true;render();});
  const options=selected.map(m=>`<option value="${esc(m)}">${esc(label(m))}</option>`).join('');$('casebankFrom').innerHTML=$('casebankTo').innerHTML=options;
  $('casebankFrom').value=selected[0];$('casebankTo').value=$('casebankPreset').value==='classic'?selected[2]:selected[1];
  $('casebankReferenceLabel').hidden=$('casebankPreset').value!=='classic';render();
 }
 function render(){if(!bank)return;const flags=getFilters(),oldPattern=$('casebankPattern').value;
  const eligible=selectCaseGroups(bank.groups,selected,{study:$('casebankStudy').value,phase:$('casebankPhase').value,
   complete:$('casebankComplete').checked,allowPartial:$('casebankAllowPartial').checked,videoOnly:flags.videoOnly,outcomes});
  const observed=[...new Set(eligible.map(g=>g.pattern))].sort();
  let patterns=selected.length===3?classicPatterns:selected.length<=4?Array.from({length:2**selected.length},(_,i)=>i.toString(2).padStart(selected.length,'0')):observed;
  patterns=[...new Set([...patterns,...observed])];const counts=Object.fromEntries(patterns.map(p=>[p,0]));for(const g of eligible)counts[g.pattern]++;
  $('casebankPattern').innerHTML='<option value="">所有成败组合</option>'+patterns.map(p=>`<option value="${p}">${esc(patternLabel(p))} (${counts[p]})</option>`).join('');
  $('casebankPattern').value=patterns.includes(oldPattern)?oldPattern:'';const pattern=$('casebankPattern').value;
  $('casebankCounts').innerHTML=patterns.slice(0,16).map(p=>`<button data-pattern="${p}" title="${esc(patternLabel(p))}" class="${pattern===p?'active':''}"><span>${esc(p.split('').map(v=>v==='1'?'成功':v==='0'?'失败':'未记录').join(' / '))}</span><strong>${counts[p]}</strong></button>`).join('');
  $('casebankCounts').querySelectorAll('button').forEach(b=>b.onclick=()=>{$('casebankPattern').value=b.dataset.pattern;render();});
  current=eligible.filter(g=>!pattern||g.pattern===pattern);
  $('casebankFlagStatus').textContent=(flags.videoOnly?'已同步左侧“仅本地有视频”；':'当前允许查看无视频旧记录；')+'成败顺序与下表方法列一致。多于4种方法时只列实际观察到的组合。';
  $('casebankCount').textContent=`${current.length}组条件 · ${new Set(current.map(g=>g.condition)).size}个不同物理状态 · ${new Set(current.map(g=>g.seed)).size}个评测seed值；同一状态的重复执行不算新场景。`;
  $('casebankHead').innerHTML='<tr><th>配对条件</th>'+selected.map(m=>`<th>${esc(label(m))}</th>`).join('')+'<th>打开回放</th></tr>';
  const cell=r=>!r?'<span class="casebank-outcome missing">未记录</span><small>未运行 / 未导入；不是失败</small>':`<span class="casebank-outcome ${r.success===true?'success':r.success===false?'failure':'missing'}">${r.success===true?'成功':r.success===false?'失败':'结局未记录'}</span><small>${r.steps??'—'}步 · 视频${r.video?(r.video_bound?'✓':'独立'):'缺'} / trace${r.trace?'✓':/^H\d+$/.test(r.method)?'不适用':'缺'} / 动作${r.actions?'✓':'缺'}</small>`;
  const from=$('casebankFrom').value,to=$('casebankTo').value;
  $('casebankRows').innerHTML=current.map((g,i)=>{const ready=from!==to&&g.methods[from]&&g.methods[to];
   const shortcuts=$('casebankPreset').value==='classic'?['H50','H20'].map(m=>`<button data-index="${i}" data-base="${m}" ${!g.methods[m]||!g.methods[selected[2]]?'disabled':''}>${m} ↔ 参照</button>`).join(''):'';
   return `<tr><td><strong>${esc(g.condition.replace('libero_object/',''))}</strong><small>seed ${g.seed} · ${esc(g.phase)}<br>${esc(studyLabel(g.study))}</small></td>${selected.map(m=>`<td>${cell(g.methods[m])}</td>`).join('')}<td><button data-index="${i}" ${ready?'':'disabled'}>打开所选两者</button>${shortcuts}<small>已记录配置 ${Object.keys(g.methods).length}</small></td></tr>`;
  }).join('')||`<tr><td colspan="${selected.length+2}">没有满足条件的真实记录。可调整来源、逐方法结局或资料筛选；缺失结果不会当作失败。</td></tr>`;
  $('casebankRows').querySelectorAll('button[data-index]').forEach(b=>b.onclick=async()=>{const g=current[Number(b.dataset.index)],a=b.dataset.base||from,z=b.dataset.base?selected[2]:to;
   if(a===z||!g.methods[a]||!g.methods[z])return;$('casebankDialog').close();try{await openPair(g.methods[a].id,g.methods[z].id);}catch(e){notify('打开对照失败：'+e.message);}});
 }
 $('casebankOpen').onclick=async()=>{try{if(!bank){bank=await api('/api/casebank');
   $('casebankReference').innerHTML=bank.reference_methods.map(r=>`<option value="${esc(r.id)}">${esc(r.label)}</option>`).join('');$('casebankReference').value='mlp_ogg';
   $('casebankStudy').innerHTML='<option value="">全部来源（每行仍独立配对）</option>'+[...new Set(bank.groups.map(g=>g.study))].map(s=>`<option value="${esc(s)}">${esc(studyLabel(s))}</option>`).join('');
   $('casebankStudy').value=bank.groups.find(g=>g.study.startsWith('replay_casebank_'))?.study||'';$('casebankComplete').checked=true;
   $('casebankMethods').innerHTML=bank.methods.map(m=>`<label><input type="checkbox" value="${esc(m.id)}">${esc(m.label)}</label>`).join('');configure(selected);
  }render();$('casebankDialog').showModal();}catch(e){notify('样本库读取失败：'+e.message);}};
 $('casebankPreset').onchange=()=>{const preset=$('casebankPreset').value;$('casebankReferenceLabel').hidden=preset!=='classic';if(preset==='custom'){$('casebankMethodDetails').open=true;return;}
  if(preset==='horizon')$('casebankStudy').value='horizon_pilot';
  configure(preset==='classic'?['H50','H20',$('casebankReference').value]:presets[preset]);};
 $('casebankReference').onchange=()=>configure(['H50','H20',$('casebankReference').value]);
 $('casebankApplyMethods').onclick=()=>{const methods=[...$('casebankMethods').querySelectorAll('input:checked')].map(e=>e.value);
  if(methods.length<2){$('casebankMethodError').textContent='至少选两种方法；未改变当前表格。';return;}$('casebankMethodError').textContent='';$('casebankPreset').value='custom';configure(methods);};
 for(const id of ['casebankStudy','casebankPhase','casebankPattern','casebankComplete','casebankAllowPartial','casebankFrom','casebankTo'])$(id).onchange=render;
 document.addEventListener('replay-filter-change',()=>{if(bank)render();});$('casebankClose').onclick=()=>$('casebankDialog').close();
}
