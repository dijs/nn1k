module.exports=function(t,e={}){let{learningRate:l=.4,momentum:n=.1,decay:o=.9999,weights:r=[],biases:i=[],activation:f=(t=>1/(1+Math.exp(-t)))}=e;const u=t.length-1,a=i.length&&r.length,s=[],c=[],h=[],g=[];let m=null;const p=t=>Array(t).fill(0),d=t=>p(t).map(t=>Math.random());for(let e=0;e<=u;e++){let l=t[e];if(c[e]=p(l),g[e]=p(l),s[e]=p(l),e>0){a||(i[e]=d(l),r[e]=new Array(l)),h[e]=new Array(l);for(let n=0;n<l;n++){let l=t[e-1];a||(r[e][n]=d(l)),h[e][n]=p(l)}}}function w(e){s[0]=e;let l=null;for(let n=1;n<=u;n++){for(let l=0;l<t[n];l++){let t=r[n][l],o=i[n][l];for(let l=0;l<t.length;l++)o+=t[l]*e[l];s[n][l]=f(o)}l=e=s[n]}return l}function y(e){w(e.input),function(e){for(let l=u;l>=0;l--)for(let n=0;n<t[l];n++){let t=s[l][n],o=0;if(l===u)o=e[n]-t;else{let t=c[l+1];for(let e=0;e<t.length;e++)o+=t[e]*r[l+1][e][n]}g[l][n]=o,c[l][n]=o*t*(1-t)}}(e.output),function(){for(let e=1;e<=u;e++){let o=s[e-1];for(let f=0;f<t[e];f++){let t=c[e][f];for(let i=0;i<o.length;i++){let u=h[e][f][i];u=l*t*o[i]+n*u,h[e][f][i]=u,r[e][f][i]+=u}i[e][f]+=l*t}}}()}return{train:function(t,e={}){const{iterations:n=2e4,stopError:r=.05,log:i,done:f=(()=>0)}=e;!function t(e,n,r,i,f){l*=o;let a=0;for(let t=0;t<e.length;t++)y(e[t]),a+=Math.abs(g[u][0]);const s=a/e.length;if(i&&n%100==0&&i(s),0==--n||s<r)return f(s);m=setTimeout(()=>t(e,n,r,i,f),0)}(t,n,r,i,f)},run:w,stop:()=>clearTimeout(m),export:()=>({weights:r,biases:i})}};