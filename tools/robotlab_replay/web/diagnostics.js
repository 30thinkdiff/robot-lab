const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

export function drawEvidence(container,summary,jump){
 container.innerHTML=summary.cards.map(c=>`<article class="evidence-card ${c.kind}"><strong>${esc(c.title)}</strong><p>${esc(c.text)}</p>${Number.isInteger(c.step)?`<button data-step="${c.step}" class="text-btn">查看 step ${c.step} ↗</button>`:''}</article>`).join('');
 container.querySelectorAll('button[data-step]').forEach(b=>b.onclick=()=>jump(Number(b.dataset.step)));
}

export function drawMatrix(container,rows,kind,jump){
 if(!rows.length){container.innerHTML='<p class="empty-detail">当前实验 / 本步未记录该数组，保持缺失。完整视觉Tensor未导入。</p>';return;}
 const columns=Math.max(...rows.map(r=>r.values.length)),cellWidth=kind==='commands'?58:18,left=66,top=27,rowHeight=19,width=left+columns*cellWidth+12,height=top+rows.length*rowHeight+12;
 const max=Math.max(...rows.flatMap(r=>r.values.map(Math.abs)),0),signed=kind==='commands';
 let content=`<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${signed?'实际提交动作分量':'OGG逐动作更新L2'}热图" style="min-width:${Math.min(width,620)}px;height:${Math.min(height,400)}px">`;
 for(let j=0;j<columns;j+=columns>20?5:1)content+=`<text x="${left+(j+.5)*cellWidth}" y="16" text-anchor="middle">${signed?'a':'slot '}${j}</text>`;
 rows.forEach((row,i)=>{
  const label=signed?'step '+row.step:'denoise '+(row.denoise??'?');content+=`<text x="${left-7}" y="${top+i*rowHeight+13}" text-anchor="end">${esc(label)}</text>`;
  row.values.forEach((v,j)=>{const a=max?Math.abs(v)/max:0,background=v===0?'#f4f7fa':signed?(v>0?`rgba(36,124,189,${.12+.88*a})`:`rgba(218,146,50,${.12+.88*a})`):`rgba(131,99,183,${.12+.88*a})`;
   content+=`<rect x="${left+j*cellWidth}" y="${top+i*rowHeight}" width="${cellWidth-1}" height="${rowHeight-1}" fill="${background}" ${signed?`data-step="${row.step}" tabindex="0" role="button" aria-label="跳转step ${row.step}，a${j}=${v}"`:''}><title>${esc(label)} · ${signed?'a':'动作slot '}${j} = ${v}</title></rect>`;
  });
 });
 content+='</svg>';
 container.innerHTML=`<p class="matrix-scale">色标：${signed?'-'+max.toPrecision(4)+' … 0 … ': '0 … '}${max.toPrecision(4)}；每张图按本窗口最大绝对值缩放，跨图请比较数值。悬停查看原值。</p><div class="matrix-scroll">${content}</div>`;
 container.querySelectorAll('[data-step]').forEach(r=>{r.onclick=()=>jump(Number(r.dataset.step));r.onkeydown=e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();jump(Number(r.dataset.step));}};});
}
