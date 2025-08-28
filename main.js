/**
 * Psychedelic Origami — Textures + Evolution + Motion
 *
 * - Texture: Perlin / fBm / Ridged / Psychedelic + Scale + Flow + Evolution (with Auto)
 * - Brightness / Contrast (each with Auto + Speed)
 * - Back-face uses same texture with +0.5 hue shift, and correct lighting
 * - Motion: Spin & Float (each with Auto + Speed)
 * - Kaleido Sectors: Auto + Speed
 * - 0..1 style sliders use decimal steps
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

/* ---------- Renderer / Scene ---------- */
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.06;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x050509, 6, 36);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 1.8, 5.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

/* ---------- Post ---------- */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.35, 0.6, 0.2);
composer.addPass(bloom);
const fxaa = new ShaderPass(FXAAShader);
fxaa.material.uniforms.resolution.value.set(1 / window.innerWidth, 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

/* ---------- Geometry ---------- */
const SIZE = 3.0;
const SEG = 160;
const sheetGeo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
sheetGeo.rotateX(-0.25);

const background = new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
);
scene.add(background);

/* ---------- Dynamic Reflection ---------- */
const cubeRT = new THREE.WebGLCubeRenderTarget(256, { generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter });
const cubeCam = new THREE.CubeCamera(0.1, 200, cubeRT);
scene.add(cubeCam);

/* ---------- Helpers ---------- */
const tmp = { v: new THREE.Vector3() };
function sdLine2(p, a, d){ const px=p.x-a.x, py=p.y-a.y; return d.x*py - d.y*px; }
function rotPointAroundAxis(p, a, axisUnit, ang){ tmp.v.copy(p).sub(a).applyAxisAngle(axisUnit, ang).add(a); p.copy(tmp.v); }
function rotVecAxis(v, axisUnit, ang){ v.applyAxisAngle(axisUnit, ang); }
function clamp(x, lo, hi){ return x<lo?lo:x>hi?hi:x; }
function clamp01(x){ return x<0?0:x>1?1:x; }

/* ---------- Minimal crease engine ---------- */
const MAX_CREASES = 6;
const MAX_MASKS_PER = 2;
const VALLEY = +1, MOUNTAIN = -1;

const base = {
  count: 0,
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  amp:  new Array(MAX_CREASES).fill(0),     // |angle| in radians
  sign: new Array(MAX_CREASES).fill(1),     // +1 valley, -1 mountain
  mCount: new Array(MAX_CREASES).fill(0),
  mA: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3())),
  mD: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3(1,0,0)))
};
function resetBase(){
  base.count=0;
  for (let i=0;i<MAX_CREASES;i++){
    base.A[i].set(0,0,0); base.D[i].set(1,0,0);
    base.amp[i]=0; base.sign[i]=1; base.mCount[i]=0;
    for (let m=0;m<MAX_MASKS_PER;m++){ base.mA[i][m].set(0,0,0); base.mD[i][m].set(1,0,0); }
  }
}
function addCrease({ Ax=0, Ay=0, Dx=1, Dy=0, deg=180, sign=VALLEY, masks=[] }){
  if (base.count >= MAX_CREASES) return;
  const i = base.count++;
  const d = new THREE.Vector2(Dx, Dy).normalize();
  base.A[i].set(Ax, Ay, 0);
  base.D[i].set(d.x, d.y, 0);
  base.amp[i]  = THREE.MathUtils.degToRad(Math.max(0, Math.min(180, Math.abs(deg))));
  base.sign[i] = sign >= 0 ? VALLEY : MOUNTAIN;
  base.mCount[i] = Math.min(MAX_MASKS_PER, masks.length);
  for (let m=0;m<base.mCount[i];m++){
    const mk = masks[m]; const dd = new THREE.Vector2(mk.Dx, mk.Dy).normalize();
    base.mA[i][m].set(mk.Ax, mk.Ay, 0);
    base.mD[i][m].set(dd.x, dd.y, 0);
  }
}

const eff = {
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  ang: new Float32Array(MAX_CREASES),
  mCount: new Int32Array(MAX_CREASES),
  mA: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3())),
  mD: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3(1,0,0)))
};

const drive = { progress:0.65, globalSpeed:1.0 };

/* ---------- Uniforms ---------- */
const uniforms = {
  uTime:          { value: 0 },
  uSectors:       { value: 10.0 },

  // texture/pattern
  uTexMode:       { value: 0 },     // 0: psychedelic, 1: perlin, 2: fbm, 3: ridged
  uTexAmt:        { value: 0.50 },  // variability/contrast inside pattern gen
  uTexSpeed:      { value: 0.60 },
  uTexScale:      { value: 1.00 },
  uTexEvo:        { value: 0.00 },  // evolution param

  // tone
  uHueShift:      { value: 0.05 },
  uTexBrightness: { value: 0.00 },  // -1..+1 add
  uTexContrast:   { value: 1.00 },  // multiply

  // paper optics
  uIridescence:   { value: 0.65 },
  uFilmIOR:       { value: 1.35 },
  uFilmNm:        { value: 360.0 },
  uFiber:         { value: 0.35 },
  uEdgeGlow:      { value: 0.7 },

  // reflections & lighting
  uEnvMap:        { value: cubeRT.texture },
  uReflectivity:  { value: 0.25 },
  uSpecIntensity: { value: 0.7 },
  uSpecPower:     { value: 24.0 },
  uRimIntensity:  { value: 0.5 },
  uLightDir:      { value: new THREE.Vector3(0.5, 1.0, 0.25).normalize() },

  // folding data
  uCreaseCount:   { value: 0 },
  uAeff:          { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3()) },
  uDeff:          { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3(1,0,0)) },
  uAng:           { value: new Float32Array(MAX_CREASES) },

  // masks (flattened)
  uMaskA:         { value: Array.from({length: MAX_CREASES*MAX_MASKS_PER}, () => new THREE.Vector3()) },
  uMaskD:         { value: Array.from({length: MAX_CREASES*MAX_MASKS_PER}, () => new THREE.Vector3(1,0,0)) },
  uMaskOn:        { value: new Float32Array(MAX_CREASES*MAX_MASKS_PER) }
};
function pushToUniforms(){
  uniforms.uCreaseCount.value = base.count;
  uniforms.uAeff.value = eff.A.map(v => v.clone());
  uniforms.uDeff.value = eff.D.map(v => v.clone());
  uniforms.uAng.value  = Float32Array.from(eff.ang);

  const flatA=[], flatD=[], on=[];
  for (let i=0;i<base.count;i++){
    for (let m=0;m<MAX_MASKS_PER;m++){
      flatA.push(eff.mA[i][m].clone()); flatD.push(eff.mD[i][m].clone());
      on.push(m < eff.mCount[i] ? 1 : 0);
    }
  }
  const pad = MAX_CREASES*MAX_MASKS_PER - flatA.length;
  for (let p=0;p<pad;p++){ flatA.push(new THREE.Vector3()); flatD.push(new THREE.Vector3(1,0,0)); on.push(0); }
  uniforms.uMaskA.value = flatA;
  uniforms.uMaskD.value = flatD;
  uniforms.uMaskOn.value = Float32Array.from(on);

  mat.uniformsNeedUpdate = true;
}

/* ---------- Shaders ---------- */
const vs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  #define MAX_MASKS_PER ${MAX_MASKS_PER}
  precision highp float;

  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];
  uniform float uAng[MAX_CREASES];
  uniform vec3  uMaskA[MAX_CREASES*MAX_MASKS_PER];
  uniform vec3  uMaskD[MAX_CREASES*MAX_MASKS_PER];
  uniform float uMaskOn[MAX_CREASES*MAX_MASKS_PER];

  varying vec3 vPos; varying vec3 vN; varying vec3 vLocal; varying vec2 vUv;

  vec3 rotAroundLine(vec3 p, vec3 a, vec3 u, float ang){
    vec3 v = p - a; float c = cos(ang), s = sin(ang);
    return a + v*c + cross(u, v)*s + u*dot(u, v)*(1.0 - c);
  }
  vec3 rotVec(vec3 v, vec3 u, float ang){ float c=cos(ang), s=sin(ang); return v*c + cross(u, v)*s + u*dot(u,v)*(1.0-c); }
  float sdLine(vec2 p, vec2 a, vec2 d){ return d.x*(p.y - a.y) - d.y*(p.x - a.x); }

  bool inMask(int i, vec2 p){
    for (int m=0; m<MAX_MASKS_PER; m++){
      int idx = i*MAX_MASKS_PER + m;
      if (uMaskOn[idx] > 0.5){
        vec2 a = uMaskA[idx].xy, d = normalize(uMaskD[idx].xy);
        if (sdLine(p, a, d) <= 0.0) return false;
      }
    }
    return true;
  }

  void main(){
    vUv = uv;
    vec3 p = position;
    vec3 n = normalize(normal);

    // sequential hinge rotations
    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;
      vec3 a = uAeff[i];
      vec3 d = normalize(uDeff[i]);
      float sd = sdLine(p.xy, a.xy, d.xy);
      if (sd > 0.0 && inMask(i, p.xy)){
        p = rotAroundLine(p, a, d, uAng[i]);
        n = normalize(rotVec(n, d, uAng[i]));
      }
    }

    vLocal = p;
    vec4 world = modelMatrix * vec4(p, 1.0);
    vPos = world.xyz;
    vN   = normalize(mat3(modelMatrix) * n);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const fs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  precision highp float;

  uniform float uTime;
  uniform float uSectors;
  uniform int   uTexMode;
  uniform float uTexAmt;
  uniform float uTexSpeed;
  uniform float uTexScale;
  uniform float uTexEvo;

  uniform float uHueShift;
  uniform float uTexBrightness;
  uniform float uTexContrast;

  uniform float uIridescence, uFilmIOR, uFilmNm, uFiber, uEdgeGlow;

  uniform samplerCube uEnvMap;
  uniform float uReflectivity;
  uniform float uSpecIntensity, uSpecPower, uRimIntensity;
  uniform vec3  uLightDir;

  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];

  varying vec3 vPos; varying vec3 vN; varying vec3 vLocal; varying vec2 vUv;

  #define PI 3.14159265359

  /* --- utility value noise for fibers etc. --- */
  float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
  float noiseVal(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=hash(i), b=hash(i+vec2(1,0)), c=hash(i+vec2(0,1)), d=hash(i+vec2(1,1));
    vec2 u=f*f*(3.0-2.0*f);
    return mix(a,b,u.x)+ (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
  }

  /* --- Perlin-style gradient noise (2D) with quintic fade --- */
  vec2 grad2(vec2 p){
    float a = 6.2831853 * hash(p);
    return vec2(cos(a), sin(a));
  }
  float fade(float t){ return t*t*t*(t*(t*6.0-15.0)+10.0); } // 6t^5 - 15t^4 + 10t^3

  float perlin(vec2 p){
    vec2 i=floor(p), f=fract(p);
    vec2 g00=grad2(i+vec2(0,0));
    vec2 g10=grad2(i+vec2(1,0));
    vec2 g01=grad2(i+vec2(0,1));
    vec2 g11=grad2(i+vec2(1,1));
    float n00=dot(g00, f-vec2(0,0));
    float n10=dot(g10, f-vec2(1,0));
    float n01=dot(g01, f-vec2(0,1));
    float n11=dot(g11, f-vec2(1,1));
    vec2 u = vec2(fade(f.x), fade(f.y));
    return mix(mix(n00,n10,u.x), mix(n01,n11,u.x), u.y);
  }
  float fbmPerlin(vec2 p){
    float sum=0.0, amp=0.5, freq=1.0;
    for(int i=0;i<6;i++){ sum += amp * perlin(p*freq); freq*=2.0; amp*=0.5; }
    return sum;
  }
  float ridged(vec2 p){
    float sum=0.0, amp=0.5, freq=1.0;
    for(int i=0;i<6;i++){
      float n = perlin(p*freq);
      n = 1.0 - abs(n);
      n *= n;
      sum += n * amp;
      freq*=2.0; amp*=0.5;
    }
    return sum;
  }

  vec3 hsv2rgb(vec3 c){
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.,4.,2.), 6.)-3.)-1., 0., 1.);
    return c.z * mix(vec3(1.0), rgb, c.y);
  }
  vec3 thinFilm(float cosTheta, float ior, float nm){
    vec3 lambda = vec3(680.0, 550.0, 440.0);
    vec3 phase  = 4.0 * PI * ior * nm * cosTheta / lambda;
    return 0.5 + 0.5*cos(phase);
  }
  float sdLine(vec2 p, vec2 a, vec2 d){ return d.x*(p.y - a.y) - d.y*(p.x - a.x); }

  /* --- Texture generators, parameterized by hue base --- */
  vec3 texPsychedelic(vec3 worldPos, float t, float hBase){
    float rScale = max(0.001, uTexScale);
    float theta = atan(worldPos.z, worldPos.x);
    float r = length(worldPos.xz) * 0.55 * rScale;
    float seg = 2.0*PI / max(3.0, uSectors);
    float aa = mod(theta, seg); aa = abs(aa - 0.5*seg);
    vec2 k = vec2(cos(aa), sin(aa)) * r;

    // domain warp using evolution
    vec2 evo = vec2(cos(uTexEvo*0.8), sin(uTexEvo*0.6)) * 0.35;
    vec2 q = (k*2.0 + vec2(0.18*t, -0.12*t)) + evo;
    q += 0.5*vec2(noiseVal(q+13.1), noiseVal(q+71.7));
    float n = noiseVal(q*2.0) * 0.75 + 0.25*noiseVal(q*5.0);
    float hue = fract(n + 0.15*sin(t*0.3) + hBase);
    float contrast = mix(1.0, 1.8, clamp(uTexAmt,0.0,1.0));
    vec3 baseCol = hsv2rgb(vec3(hue, 0.9, smoothstep(0.25,1.0,n)));
    return pow(baseCol, vec3(contrast));
  }

  vec3 texPerlin(vec2 p, float t, float hBase){
    p = p * max(0.001, uTexScale);
    // evolution as domain warp
    p += 0.35*vec2(cos(uTexEvo*0.9 + p.y*0.2), sin(uTexEvo*0.7 + p.x*0.2));
    p += vec2(0.17*t, -0.11*t);
    float n = perlin(p);
    float k = pow(0.5*(n+1.0), mix(1.0, 3.0, clamp(uTexAmt,0.0,1.0)));
    vec3 a = hsv2rgb(vec3(fract(hBase), 0.85, 0.9));
    vec3 b = hsv2rgb(vec3(fract(hBase + 0.25), 0.9, 0.95));
    return mix(a, b, k);
  }

  vec3 texFBM(vec2 p, float t, float hBase){
    p = p * max(0.001, uTexScale);
    p += 0.35*vec2(cos(uTexEvo*0.8 + p.y*0.15), sin(uTexEvo*0.5 + p.x*0.15));
    p += vec2(0.15*t, -0.09*t);
    float f = fbmPerlin(p);
    float k = pow(0.5*(f+1.0), mix(1.0, 3.0, clamp(uTexAmt,0.0,1.0)));
    vec3 a = hsv2rgb(vec3(fract(hBase + 0.05), 0.8, 0.92));
    vec3 b = hsv2rgb(vec3(fract(hBase + 0.35), 0.9, 0.95));
    return mix(a, b, k);
  }

  vec3 texRidged(vec2 p, float t, float hBase){
    p = p * max(0.001, uTexScale);
    p += 0.35*vec2(cos(uTexEvo*0.6 + p.y*0.1), sin(uTexEvo*0.4 + p.x*0.1));
    p += vec2(0.12*t, 0.08*t);
    float r = ridged(p);
    float k = pow(clamp(r,0.0,1.0), mix(1.0, 2.5, clamp(uTexAmt,0.0,1.0)));
    vec3 a = hsv2rgb(vec3(fract(hBase + 0.10), 0.85, 0.9));
    vec3 b = hsv2rgb(vec3(fract(hBase + 0.55), 0.9, 0.95));
    return mix(a, b, k);
  }

  void main(){
    float tTex = uTime * uTexSpeed + uTexEvo;

    // choose hue base; back-face gets +0.5 hue shift
    float hueBase = fract(uHueShift + (gl_FrontFacing ? 0.0 : 0.5));

    // choose base color by texture mode
    vec3 baseCol;
    if (uTexMode == 0){
      baseCol = texPsychedelic(vPos, tTex, hueBase);
    } else if (uTexMode == 1){
      baseCol = texPerlin(vLocal.xy, tTex, hueBase);
    } else if (uTexMode == 2){
      baseCol = texFBM(vLocal.xy, tTex, hueBase);
    } else {
      baseCol = texRidged(vLocal.xy, tTex, hueBase);
    }

    // apply texture tone (contrast & brightness)
    baseCol = baseCol * uTexContrast + vec3(uTexBrightness);
    baseCol = clamp(baseCol, 0.0, 1.0);

    // paper fibers
    float fiberLines = 0.0;
    {
      float warp = 0.0;
      vec2 pp = vLocal.xy;
      for(int i=0;i<2;i++){ warp += noiseVal(pp*4.0 + vec2(0.2*uTime, -0.1*uTime)); pp*=1.8; }
      float l = sin(vLocal.y*420.0 + warp*8.0);
      float widthAA = fwidth(l);
      fiberLines = smoothstep(0.6, 0.6 - widthAA, abs(l));
    }
    float grain = 0.0; { grain = noiseVal(vLocal.xy*25.0); }
    baseCol *= 1.0 + uFiber*(0.06*grain - 0.03) + uFiber*0.08*fiberLines;

    // distance to nearest crease for glow
    float minD = 1e9;
    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;
      vec2 a2 = uAeff[i].xy;
      vec2 d2 = normalize(uDeff[i].xy);
      float sd = abs(sdLine(vLocal.xy, a2, d2));
      minD = min(minD, sd);
    }
    float aa = fwidth(minD);
    float edge = 1.0 - smoothstep(0.0025, 0.0025 + aa, minD);

    // view/lighting
    vec3 V = normalize(cameraPosition - vPos);
    // Fix lighting on back faces: flip normal if back-facing
    vec3 N = normalize(vN) * (gl_FrontFacing ? 1.0 : -1.0);

    float cosT = clamp(dot(N, V), 0.0, 1.0);
    vec3 film = thinFilm(cosT, uFilmIOR, uFilmNm);
    float F = pow(1.0 - cosT, 5.0);
    vec3 col = mix(baseCol, mix(baseCol, film, uIridescence), F);

    vec3 L = normalize(uLightDir);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), uSpecPower) * uSpecIntensity;
    float rim  = pow(1.0 - max(dot(N, V), 0.0), 2.0) * uRimIntensity;
    col += spec + rim;

    vec3 R = reflect(-V, N);
    vec3 env = textureCube(uEnvMap, R).rgb;
    col = mix(col, env, clamp(uReflectivity, 0.0, 1.0));

    // crease glow tint
    col += uEdgeGlow * edge * film * 0.6;

    float vign = smoothstep(1.2, 0.2, length(vUv-0.5)*1.2);
    gl_FragColor = vec4(col*vign, 1.0);
  }
`;

/* ---------- Material + Mesh ---------- */
const mat = new THREE.ShaderMaterial({
  vertexShader: vs, fragmentShader: fs, uniforms,
  side: THREE.DoubleSide, extensions: { derivatives: true }
});
const sheet = new THREE.Mesh(sheetGeo, mat);
scene.add(sheet);

/* ---------- Presets ---------- */
function preset_half_vertical_valley(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
}
function preset_diagonal_valley(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:1, Dy:1, deg:180, sign:VALLEY });
}
function preset_gate_valley(){
  resetBase();
  const x = SIZE*0.25;
  addCrease({ Ax:+x, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
  addCrease({ Ax:-x, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
}

/* ---------- DOM ---------- */
const el = (id)=>document.getElementById(id);

// Toolbar
const presetSel   = el('preset');
const btnSnap     = el('btnSnap');
const autoFx      = el('autoFx');
const progress    = el('progress');
const progressOut = el('progressOut');
const globalSpeed = el('globalSpeed');
const globalSpeedOut = el('globalSpeedOut');
const btnMore     = el('btnMore');
const drawer      = el('drawer');

// Pattern
const sectors     = el('sectors');
const sectorsOut  = el('sectorsOut');
const sectorsAuto = el('sectorsAuto');
const sectorsSpeed= el('sectorsSpeed');
const sectorsSpeedOut = el('sectorsSpeedOut');

// Texture core
const texMode     = el('texMode');
const texAmp      = el('texAmp');
const texAmpOut   = el('texAmpOut');
const texSpeed    = el('texSpeed');
const texSpeedOut = el('texSpeedOut');
const texScale    = el('texScale');
const texScaleOut = el('texScaleOut');

// Evolution
const texEvo      = el('texEvo');
const texEvoOut   = el('texEvoOut');
const texEvoSpeed = el('texEvoSpeed');
const texEvoSpeedOut = el('texEvoSpeedOut');
const texEvoAuto  = el('texEvoAuto');

// Tone
const brightAmt      = el('brightAmt');
const brightAmtOut   = el('brightAmtOut');
const brightSpeed    = el('brightSpeed');
const brightSpeedOut = el('brightSpeedOut');
const brightAuto     = el('brightAuto');

const contrAmt      = el('contrAmt');
const contrAmtOut   = el('contrAmtOut');
const contrSpeed    = el('contrSpeed');
const contrSpeedOut = el('contrSpeedOut');
const contrAuto     = el('contrAuto');

// Motion
const spinAngle      = el('spinAngle');
const spinAngleOut   = el('spinAngleOut');
const spinSpeed      = el('spinSpeed');
const spinSpeedOut   = el('spinSpeedOut');
const spinAuto       = el('spinAuto');

const floatAmp       = el('floatAmp');
const floatAmpOut    = el('floatAmpOut');
const floatSpeed     = el('floatSpeed');
const floatSpeedOut  = el('floatSpeedOut');
const floatAuto      = el('floatAuto');

/* ---------- UI wiring ---------- */
function bindRangeWithOut(rangeEl, outEl, decimals=2, onInput){
  const sync = ()=>{ outEl.textContent = Number(rangeEl.value).toFixed(decimals); if(onInput) onInput(parseFloat(rangeEl.value)); };
  rangeEl.addEventListener('input', sync); sync();
}

bindRangeWithOut(progress,    progressOut,    3, v => { drive.progress = v; });
bindRangeWithOut(globalSpeed, globalSpeedOut, 2, v => { drive.globalSpeed = v; });

bindRangeWithOut(sectors,     sectorsOut,     0, v => { if(!sectorsAuto.checked) uniforms.uSectors.value = v; });
bindRangeWithOut(sectorsSpeed, sectorsSpeedOut, 2);

bindRangeWithOut(texAmp,      texAmpOut,      2, v => { if(!autoFx.checked) uniforms.uTexAmt.value = v; });
bindRangeWithOut(texSpeed,    texSpeedOut,    2);
bindRangeWithOut(texScale,    texScaleOut,    2, v => { uniforms.uTexScale.value = v; });

bindRangeWithOut(texEvo,        texEvoOut,        2, v => { if(!texEvoAuto.checked) uniforms.uTexEvo.value = v; });
bindRangeWithOut(texEvoSpeed,   texEvoSpeedOut,   2);

bindRangeWithOut(brightAmt,     brightAmtOut,     2, v => { if(!brightAuto.checked) uniforms.uTexBrightness.value = v; });
bindRangeWithOut(brightSpeed,   brightSpeedOut,   2);

bindRangeWithOut(contrAmt,      contrAmtOut,      2, v => { if(!contrAuto.checked) uniforms.uTexContrast.value = 1.0 + v; });
bindRangeWithOut(contrSpeed,    contrSpeedOut,    2);

bindRangeWithOut(spinAngle,     spinAngleOut,     1);
bindRangeWithOut(spinSpeed,     spinSpeedOut,     1);
bindRangeWithOut(floatAmp,      floatAmpOut,      2);
bindRangeWithOut(floatSpeed,    floatSpeedOut,    2);

texMode.addEventListener('change', () => {
  uniforms.uTexMode.value = parseInt(texMode.value, 10) | 0;
});

btnMore.addEventListener('click', () => {
  drawer.open = !drawer.open;
  btnMore.setAttribute('aria-expanded', drawer.open ? 'true' : 'false');
});

btnSnap.addEventListener('click', () => {
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'origami.png'; a.click();
    URL.revokeObjectURL(url);
  }, 'image/png', 1.0);
});

presetSel.addEventListener('change', () => {
  const v = presetSel.value;
  if (v==='half-vertical-valley') preset_half_vertical_valley();
  else if (v==='diagonal-valley') preset_diagonal_valley();
  else if (v==='gate-valley') preset_gate_valley();
  // subtle feedback
  camera.position.x += (Math.random()-0.5)*0.02;
  camera.position.y += (Math.random()-0.5)*0.02;
});

/* ---------- Animated parameters ---------- */
const P = {
  // earlier visuals respect master Auto FX
  hue:  { base:0.05, min:0,   max:1,   phase:0.00, ampEl: el('hueAmp'),  spdEl: el('hueSpeed'),  set:v=>uniforms.uHueShift.value=v },
  film: { base:360,  min:100, max:800, phase:0.65, ampEl: el('filmAmp'), spdEl: el('filmSpeed'), set:v=>uniforms.uFilmNm.value=v },
  edge: { base:0.70, min:0,   max:2,   phase:1.30, ampEl: el('edgeAmp'), spdEl: el('edgeSpeed'), set:v=>uniforms.uEdgeGlow.value=v },
  refl: { base:0.25, min:0,   max:1,   phase:2.10, ampEl: el('reflAmp'), spdEl: el('reflSpeed'), set:v=>uniforms.uReflectivity.value=v },
  spec: { base:0.70, min:0,   max:2,   phase:0.25, ampEl: el('specAmp'), spdEl: el('specSpeed'), set:v=>uniforms.uSpecIntensity.value=v },
  rim:  { base:0.50, min:0,   max:2,   phase:0.85, ampEl: el('rimAmp'),  spdEl: el('rimSpeed'),  set:v=>uniforms.uRimIntensity.value=v },
  bstr: { base:0.35, min:0,   max:2.5, phase:1.75, ampEl: el('bloomAmp'), spdEl: el('bloomSpeed'), set:v=>{ bloom.strength=v; } },
  brad: { base:0.60, min:0,   max:1.5, phase:2.50, ampEl: el('bloomRadAmp'), spdEl: el('bloomRadSpeed'), set:v=>{ bloom.radius=v; } },
  texAmt: { base:0.50, min:0, max:1, phase:1.10, ampEl: texAmp, spdEl: texSpeed, set:v=>uniforms.uTexAmt.value=v },
};

function osc(base, amp, w, t){ return base + amp * Math.sin(w * t); }
function applyAnimatedParam(def, t){
  const amp = +def.ampEl.value;
  const spd = +def.spdEl.value;
  const g   = drive.globalSpeed;
  const val = autoFx.checked ? osc(def.base, amp, g*spd, t + def.phase) : clamp(def.base + amp, def.min, def.max);
  def.set(clamp(val, def.min, def.max));
}

/* Per-parameter autos (not governed by Auto FX) */
function applyAutoBrightness(t){
  const g = drive.globalSpeed;
  const amp = +brightAmt.value; const spd = +brightSpeed.value;
  const val = brightAuto.checked ? osc(0.0, amp, g*spd, t) : (0.0 + amp);
  uniforms.uTexBrightness.value = clamp(val, -1.0, 1.0);
}
function applyAutoContrast(t){
  const g = drive.globalSpeed;
  const amp = +contrAmt.value; const spd = +contrSpeed.value;
  const val = contrAuto.checked ? osc(1.0, amp, g*spd, t) : (1.0 + amp);
  uniforms.uTexContrast.value = clamp(val, 0.0, 3.0);
}
function applyAutoEvolution(t){
  const g = drive.globalSpeed;
  const base = +texEvo.value;
  const spd = +texEvoSpeed.value;
  const val = texEvoAuto.checked ? (base + g*spd*t) : base;
  uniforms.uTexEvo.value = val;
}
function applyAutoSectors(t){
  if (!sectorsAuto.checked) return;
  const g = drive.globalSpeed;
  const spd = +sectorsSpeed.value;
  const s = Math.round(THREE.MathUtils.mapLinear(Math.sin(g*spd*t)*0.5+0.5, 0, 1, 3, 24));
  uniforms.uSectors.value = s;
  sectorsOut.textContent = String(s);
}

/* Motion */
function applyMotion(t){
  const g = drive.globalSpeed;
  const angBase = THREE.MathUtils.degToRad(+spinAngle.value);
  const angSpd  = THREE.MathUtils.degToRad(+spinSpeed.value);
  const angle   = spinAuto.checked ? (angBase + angSpd*g*t) : angBase;
  sheet.rotation.y = angle;

  const amp = +floatAmp.value;
  const spd = +floatSpeed.value;
  const y = floatAuto.checked ? (amp*Math.sin(g*spd*t*1.0)) : amp;
  sheet.position.y = y;
}

/* ---------- Folding ---------- */
function computeAngles(){
  const t = clamp01(drive.progress);
  for (let i=0;i<base.count;i++){
    eff.ang[i] = base.sign[i] * base.amp[i] * t;
    eff.mCount[i] = base.mCount[i];
    for (let m=0;m<MAX_MASKS_PER;m++){
      eff.mA[i][m].copy(base.mA[i][m]);
      eff.mD[i][m].copy(base.mD[i][m]);
    }
  }
  for (let i=base.count;i<MAX_CREASES;i++){ eff.ang[i]=0; eff.mCount[i]=0; }
}
function computeFrames(){
  for (let i=0;i<base.count;i++){
    eff.A[i].copy(base.A[i]); eff.D[i].copy(base.D[i]).normalize();
  }
  for (let j=0;j<base.count;j++){
    const Aj = eff.A[j]; const Dj = eff.D[j].clone().normalize(); const ang = eff.ang[j]; if (Math.abs(ang)<1e-7) continue;
    for (let k=j+1;k<base.count;k++){
      const sd = sdLine2(eff.A[k], Aj, Dj);
      if (sd > 0.0){
        rotPointAroundAxis(eff.A[k], Aj, Dj, ang);
        rotVecAxis(eff.D[k], Dj, ang); eff.D[k].normalize();
        for (let m=0;m<MAX_MASKS_PER;m++){
          const sdM = sdLine2(eff.mA[k][m], Aj, Dj);
          if (sdM > 0.0){
            rotPointAroundAxis(eff.mA[k][m], Aj, Dj, ang);
            rotVecAxis(eff.mD[k][m], Dj, ang); eff.mD[k][m].normalize();
          }
        }
      }
    }
  }
}
function pushAll(){ pushToUniforms(); }

/* ---------- Start ---------- */
preset_half_vertical_valley();
presetSel.value = 'half-vertical-valley';
uniforms.uTexMode.value = parseInt(texMode.value, 10) | 0;

/* ---------- Frame loop ---------- */
function tick(tMs){
  const t = (tMs * 0.001);
  uniforms.uTime.value = t;

  computeAngles();
  computeFrames();
  pushAll();

  // Auto FX group
  applyAnimatedParam(P.hue,  t);
  applyAnimatedParam(P.film, t);
  applyAnimatedParam(P.edge, t);
  applyAnimatedParam(P.refl, t);
  applyAnimatedParam(P.spec, t);
  applyAnimatedParam(P.rim,  t);
  applyAnimatedParam(P.bstr, t);
  applyAnimatedParam(P.brad, t);
  applyAnimatedParam(P.texAmt, t);

  // Per-parameter autos
  uniforms.uTexSpeed.value = drive.globalSpeed * (+texSpeed.value);
  applyAutoEvolution(t);
  applyAutoBrightness(t);
  applyAutoContrast(t);
  applyAutoSectors(t);
  uniforms.uTexScale.value = +texScale.value;

  // Motion
  applyMotion(t);

  // reflections (hide sheet to avoid self-reflection)
  sheet.visible = false;
  cubeCam.position.copy(sheet.position);
  cubeCam.update(renderer, scene);
  sheet.visible = true;

  controls.update();
  composer.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

/* ---------- Resize ---------- */
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h); composer.setSize(w, h);
  fxaa.material.uniforms.resolution.value.set(1 / w, 1 / h);
});
