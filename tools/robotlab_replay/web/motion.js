// Motion is a separately verified optional recording, never synthesized from u_t.
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const fmt=(v,n=3)=>finite(v)?v.toFixed(n):'未记录';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors=['#189a90','#d69232','#8f76c5'];
let last,projection='xy',view='tracking';
const $=id=>document.getElementById(id);

function trajectory(rows,step){
 const axes=projection==='xy'?[0,1]:[0,2],names=projection==='xy'?['X','Y']:['X','Z'];
 const points=rows.flatMap(r=>[r.before.eef_position_m,r.after.eef_position_m,r.nominal.position_m,...(r.goals||[]).map(g=>g.position_m)]);
 const ranges=axes.map(a=>{const v=points.map(p=>p[a]);const lo=Math.min(...v),hi=Math.max(...v),pad=Math.max(.008,(hi-lo)*.08);return[lo-pad,hi+pad];});
 const X=p=>48+(p[axes[0]]-ranges[0][0])/(ranges[0][1]-ranges[0][0])*540;
 const Y=p=>228-(p[axes[1]]-ranges[1][0])/(ranges[1][1]-ranges[1][0])*200;
 let s='<svg viewBox="0 0 620 270" role="img" aria-label="实测与名义末端路径投影">';
 for(let i=0;i<5;i++){const x=48+i*135,y=28+i*50;s+=`<path d="M${x} 28 V228 M48 ${y} H588" stroke="#e7edf3"/><text x="${x}" y="248" text-anchor="middle">${(ranges[0][0]+i/4*(ranges[0][1]-ranges[0][0])).toFixed(2)}</text><text x="43" y="${y+4}" text-anchor="end">${(ranges[1][1]-i/4*(ranges[1][1]-ranges[1][0])).toFixed(2)}</text>`;}
 const path=(list,color,dash='')=>`<path d="${list.map((p,i)=>(i?'L':'M')+X(p).toFixed(2)+','+Y(p).toFixed(2)).join(' ')}" fill="none" stroke="${color}" stroke-width="2" ${dash}/>`;
 s+=path([...rows.map(r=>r.before.eef_position_m),rows.at(-1).after.eef_position_m],colors[0]);
 // Reset nominal curves at each actual plan boundary; never join distinct plans.
 for(let start=0;start<rows.length;){let end=start+1;while(end<rows.length&&rows[end].chunk_id===rows[start].chunk_id)end++;s+=path([rows[start].before.eef_position_m,...rows.slice(start,end).map(r=>r.nominal.position_m)],colors[1],'stroke-dasharray="5 4"');start=end;}
 const r=rows[step];for(const [p,c]of[[r.before.eef_position_m,colors[0]],[r.after.eef_position_m,'#1e4058'],[r.nominal.position_m,colors[1]],...(r.goals||[]).map(g=>[g.position_m,colors[2]])])s+=`<circle cx="${X(p)}" cy="${Y(p)}" r="4" fill="${c}"/>`;
 return s+`<text x="300" y="266">${names[0]} / m</text><text x="4" y="15">${names[1]} / m</text></svg>`;
}
function series(rows,step){
 const configs={tracking:[['OSC目标误差 / mm',r=>r.tracking_error?.position_m*1000],['块名义误差 / mm',r=>r.nominal_block_error?.position_m*1000]],rotation:[['OSC朝向误差 / °',r=>r.tracking_error?.orientation_rad*180/Math.PI],['块名义朝向误差 / °',r=>r.nominal_block_error?.orientation_rad*180/Math.PI]],torque:[['命令峰值L2 / N·m',r=>r.peak_command_torque_l2]]};
 const lines=configs[view],max=Math.max(.001,...rows.flatMap(r=>lines.map(([,f])=>f(r)).filter(finite)))*1.1;
 const X=i=>45+i/Math.max(1,rows.length-1)*545,Y=v=>158-v/max*132;
 let s='<svg viewBox="0 0 620 195" role="img" aria-label="控制误差与力矩环境步曲线">';
 for(let i=0;i<4;i++)s+=`<path d="M45 ${Y(max*i/3)} H590" stroke="#e7edf3"/><text x="40" y="${Y(max*i/3)+3}" text-anchor="end">${(max*i/3).toFixed(1)}</text>`;
 lines.forEach(([label,f],k)=>{let path='',connected=false;for(const r of rows){const v=f(r);if(!finite(v)){connected=false;continue;}path+=(connected?' L':' M')+X(r.step).toFixed(2)+','+Y(v).toFixed(2);connected=true;}s+=`<path d="${path}" stroke="${colors[k]}" stroke-width="1.8" fill="none"/><text x="${45+k*260}" y="13" fill="${colors[k]}">${label}</text>`;});
 return s+`<path d="M${X(step)} 22 V160" stroke="#1e4058"/><text x="45" y="179">step 0</text><text x="590" y="179" text-anchor="end">step ${rows.length-1}</text><text x="260" y="193">环境步（本步执行后误差）</text></svg>`;
}
export function renderMotion(detail,step,jump,side='A',paired=false){
 last={detail,step,jump,side,paired};const m=detail?.motion,r=m?.rows?.[step];
 $('motionLane').hidden=!paired;$('motionLane').value=side==='B'?'b':'a';
 $('motionPosition').textContent=detail?`${detail.run.method} · step ${step}`:'尚未选择回合';
 $('motionContent').hidden=!m?.available||!r;$('motionMissing').hidden=!!m?.available&&!!r;
 $('motionMissing').textContent=!m?.available?(m?.message||'本回合未记录运动状态。'): '超出本回合记录范围，不沿用末次状态。';
 if(!m?.available||!r)return;
 $('motionPhase').textContent=m.phase;
 $('motionSummary').innerHTML=[['本步OSC位置误差',fmt(r.tracking_error?.position_m*1000,2)+' mm','真实OSC位置目标与本步执行后位置的欧氏距离；不是VLA规划好坏评分。'],['本步OSC朝向误差',fmt(r.tracking_error?.orientation_rad*180/Math.PI,2)+' °','目标与执行后末端朝向之间的最短旋转角，不是某个关节角误差。'],['相对动作块名义端点',fmt(r.nominal_block_error?.position_m*1000,2)+' mm','从本动作块起点按缩放后的动作累计理想跟踪端点，与执行后实际位置比较；可能累积，换块重置。'],['控制器更新 / 力矩限幅',`${r.controller_updates} / ${r.clipped_updates}`,'本控制周期内部力矩更新次数 / 其中至少一个关节力矩被限幅的次数；不是VLA前向次数或输入动作裁剪次数。']].map(([a,b,help])=>`<div><span title="${esc(help)}">${a} ⓘ</span><strong>${b}</strong></div>`).join('');
 const vec=v=>Array.isArray(v)?v.map(x=>fmt(x,4)).join(' · '):'未记录';
 $('motionPose').innerHTML='<table><thead><tr><th>末端位置 / m</th><th>X · Y · Z</th></tr></thead><tbody>'+[['视频时刻 Zₙ（执行前）',r.before.eef_position_m],['OSC真实目标（本步）',r.goals?.[0]?.position_m],['本步执行后 Zₙ₊₁',r.after.eef_position_m],['块名义端点（理想跟踪解释）',r.nominal.position_m]].map(([a,b])=>`<tr><td>${a}</td><td><code>${vec(b)}</code></td></tr>`).join('')+'</tbody></table>';
 $('motionCommands').innerHTML='<table><thead><tr><th>动作维度</th><th>选中归一化动作</th><th>提交给控制器</th></tr></thead><tbody>'+['平移 X','平移 Y','平移 Z','旋转 X','旋转 Y','旋转 Z','夹爪'].map((name,i)=>`<tr><td>${name}</td><td>${fmt(r.selected_normalized_action?.[i],4)}</td><td>${fmt(r.submitted_command?.[i],4)}</td></tr>`).join('')+'</tbody></table><p class="motion-limits">控制器输入先经过冻结缩放/裁剪再成为末端增量；这里的数值不是位置m或角度rad。</p>';
 $('motionTrajectory').innerHTML=trajectory(m.rows,step);$('motionSeries').innerHTML=series(m.rows,step);
 $('motionSeries').onclick=e=>{const rect=$('motionSeries').querySelector('svg').getBoundingClientRect();jump(Math.max(0,Math.min(m.rows.length-1,Math.round(((e.clientX-rect.left)/rect.width*620-45)/545*(m.rows.length-1)))));};
 $('motionJoints').innerHTML='<table><thead><tr><th>关节</th><th>执行前 / rad</th><th>执行后 / rad</th><th>速度 / rad·s⁻¹</th><th>末次力矩命令 / N·m</th></tr></thead><tbody>'+r.before.joint_position_rad.map((v,i)=>`<tr><td>J${i+1}</td><td>${fmt(v,4)}</td><td>${fmt(r.after.joint_position_rad[i],4)}</td><td>${fmt(r.after.joint_velocity_rad_s?.[i],4)}</td><td>${fmt(r.last_torque?.post_limit_torque?.[i],3)}</td></tr>`).join('')+'</tbody></table>';
 $('motionRaw').textContent=JSON.stringify(r,null,2);
 $('motionLimits').textContent=m.message+' '+m.nominal_limit+' OSC_POSE直接计算力矩，没有逐步关节目标角；末次力矩命令与动力学实际作用力分别保留。';
}
document.getElementById('motionProjection').onchange=e=>{projection=e.target.value;if(last)renderMotion(last.detail,last.step,last.jump,last.side,last.paired);};
document.getElementById('motionView').onchange=e=>{view=e.target.value;if(last)renderMotion(last.detail,last.step,last.jump,last.side,last.paired);};
