const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const patterns=['001','000','011','111','010','101','100','110'];
const patternLabel=p=>`H50${p[0]==='1'?'成功':'失败'} · H20${p[1]==='1'?'成功':'失败'} · 参照${p[2]==='1'?'成功':'失败'}`;

export function setupCasebank({api,openPair,notify}){
 let bank=null,current=[];
 const references=()=>bank?.reference_methods||[];
 function render(){if(!bank)return;const reference=$('casebankReference').value,study=$('casebankStudy').value,phase=$('casebankPhase').value,pattern=$('casebankPattern').value;
  const eligible=bank.groups.filter(g=>(!study||g.study===study)&&(!phase||g.phase===phase)&&['H50','H20',reference].every(m=>g.methods[m])&&(!$('casebankComplete').checked||['H50','H20',reference].every(m=>g.methods[m].replay_complete)));
  const counts=Object.fromEntries(patterns.map(p=>[p,0]));
  for(const g of eligible){g.pattern=['H50','H20',reference].map(m=>g.methods[m].success?'1':'0').join('');counts[g.pattern]++;}
  $('casebankCounts').innerHTML=patterns.map(p=>`<button data-pattern="${p}" class="${pattern===p?'active':''}">${esc(patternLabel(p))}<strong>${counts[p]}</strong></button>`).join('');
  $('casebankCounts').querySelectorAll('button').forEach(b=>b.onclick=()=>{$('casebankPattern').value=b.dataset.pattern;render();});
  current=eligible.filter(g=>!pattern||g.pattern===pattern);
  $('casebankCount').textContent=`${current.length}组配对 · ${new Set(current.map(g=>g.study+':'+g.condition)).size}个实验/物理状态；每行共享task / init / 评测seed`;
  const cell=r=>`<span class="casebank-outcome ${r.success?'success':'failure'}">${r.success?'成功':'失败'}</span><small>${r.steps}步 · 视频${r.video?'✓':'缺'} / trace${r.trace?'✓':r.method.startsWith('H')?'不适用':'缺'} / 动作${r.actions?'✓':'缺'}</small>`;
  $('casebankRows').innerHTML=current.map((g,i)=>`<tr><td><strong>${esc(g.condition.replace('libero_object/',''))}</strong><small>seed ${g.seed} · ${esc(g.phase)}<br>${g.study==='legacy_five_models'?'旧正式五模型':esc(g.study)}</small></td><td>${cell(g.methods.H50)}</td><td>${cell(g.methods.H20)}</td><td>${cell(g.methods[reference])}</td><td><button data-index="${i}" data-base="H50">H50 ↔ 参照</button><button data-index="${i}" data-base="H20">H20 ↔ 参照</button><small>已记录配置 ${Object.keys(g.methods).length} · 同状态seed ${eligible.filter(x=>x.study===g.study&&x.condition===g.condition).length}</small></td></tr>`).join('')||'<tr><td colspan="5">没有满足条件的真实记录。缺失组合不补造；补测结果导入后重启Replay即可索引。</td></tr>';
  $('casebankRows').querySelectorAll('button[data-index]').forEach(b=>b.onclick=async()=>{const g=current[Number(b.dataset.index)];$('casebankDialog').close();try{await openPair(g.methods[b.dataset.base].id,g.methods[reference].id);}catch(e){notify('打开配对失败：'+e.message);}});
 }
 $('casebankOpen').onclick=async()=>{try{if(!bank){bank=await api('/api/casebank');$('casebankReference').innerHTML=references().map(r=>`<option value="${esc(r.id)}">${esc(r.label)}</option>`).join('');$('casebankReference').value='mlp_ogg';$('casebankStudy').innerHTML='<option value="">全部来源</option>'+[...new Set(bank.groups.map(g=>g.study))].map(s=>`<option value="${esc(s)}">${s==='legacy_five_models'?'旧正式五模型筛选库':esc(s)}</option>`).join('');$('casebankPattern').innerHTML='<option value="">所有成败组合</option>'+patterns.map(p=>`<option value="${p}">${esc(patternLabel(p))}</option>`).join('');}
   render();$('casebankDialog').showModal();}catch(e){notify('样本库读取失败：'+e.message);}};
 for(const id of ['casebankReference','casebankStudy','casebankPhase','casebankPattern','casebankComplete'])$(id).onchange=render;
 $('casebankClose').onclick=()=>$('casebankDialog').close();
}
