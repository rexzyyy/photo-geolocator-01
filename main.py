#!/usr/bin/env python3
"""Model-free photo geolocator: JPEG DC colour cues and broad regional priors."""
import json, math, os, struct, urllib.request

FEATS = ['bluesky','skyBlue','skySat','midB','B','midGreen','midG','gray','gndB','yellow','white','gndG','gndR','gndRed','lrgrad','Ysd']
MEANS = [0.314,53.074,54.184,128.298,129.422,16.863,124.34,0.156,93.788,0.004,0.074,99.676,94.938,-4.738,1.372,58.526]
SDS = [0.255,46.266,45.14,43.44,27.526,20.781,33.521,0.136,44.216,0.015,0.1,47.782,48.582,14.744,32.331,12.834]
# Aggregated regional centres and feature profiles; each profile is a regional mean.
ANCHORS = json.loads('''[[-37.68,-67.81,[0.31,60.51,60.51,149.63,155.65,38.48,143.59,0.19,141.57,0.01,0.09,170.0,175.13,5.14,-7.96,40.65]],[-44.16,169.86,[0.33,64.94,64.94,76.29,102.62,11.35,76.13,0.11,74.25,0.0,0.06,76.8,69.52,-7.28,-23.52,70.25]],[-25.05,-67.41,[0.27,84.85,84.85,117.98,140.4,-0.16,126.67,0.11,133.65,0.0,0.0,144.59,141.77,-2.82,8.97,41.97]],[-19.29,-45.41,[0.15,34.16,34.29,110.46,118.43,11.95,110.35,0.17,71.07,0.01,0.14,79.76,77.43,-2.33,-4.19,63.96]],[-13.48,40.34,[0.47,137.83,137.83,178.56,165.35,66.22,159.1,0.05,120.83,0.02,0.04,142.84,133.4,-9.44,-9.26,47.59]],[-15.59,147.89,[0.35,72.03,72.03,97.83,112.21,14.97,111.91,0.04,69.52,0.02,0.01,88.03,84.72,-3.31,-9.58,57.96]],[22.37,-96.98,[0.31,27.74,29.85,158.43,129.52,11.83,151.95,0.2,68.47,0.0,0.08,68.59,66.98,-1.62,-30.04,63.64]],[7.9,-6.6,[0.35,61.31,61.31,125.08,118.09,26.61,112.55,0.05,60.99,0.0,0.03,66.21,58.57,-7.64,16.27,63.47]],[8.22,34.62,[0.39,47.12,50.96,106.74,133.51,0.16,112.7,0.04,139.84,0.03,0.09,141.6,145.28,3.68,-11.76,56.71]],[15.13,87.64,[0.18,24.37,29.6,107.37,115.09,10.35,110.12,0.22,100.58,0.0,0.06,105.53,102.51,-3.02,4.26,58.82]],[50.17,-124.14,[0.34,41.03,42.54,123.82,99.52,5.01,112.88,0.11,48.24,0.0,0.02,42.72,41.09,-1.63,35.11,53.83]],[42.57,-94.31,[0.4,62.94,64.1,144.71,149.52,24.75,138.1,0.15,136.77,0.0,0.01,144.57,138.86,-5.71,13.21,47.14]],[46.19,33.4,[0.27,45.51,45.85,139.12,137.54,12.85,137.62,0.21,99.86,0.0,0.12,100.79,92.91,-7.88,8.22,64.96]],[46.8,71.19,[0.18,22.2,22.62,106.49,114.95,7.55,107.25,0.27,73.26,0.0,0.12,81.05,79.05,-2.0,3.91,63.35]],[40.47,133.83,[0.27,44.47,44.9,120.77,116.07,10.0,118.19,0.23,78.01,0.0,0.08,76.29,71.86,-4.42,-19.46,64.88]],[62.23,-136.51,[0.74,144.32,144.32,213.75,178.61,50.56,168.89,0.03,95.85,0.0,0.0,106.6,111.39,4.8,9.98,53.31]],[67.01,22.46,[0.5,63.45,64.56,122.55,122.61,17.42,110.24,0.09,77.71,0.0,0.05,79.25,64.71,-14.55,14.59,54.48]]]''')
CENTROID=(22.17,10.51)

class Bits:
    def __init__(self, data, offset): self.d=data; self.p=offset; self.buf=0; self.n=0
    def get(self, count):
        while self.n < count:
            if self.p >= len(self.d): b=255
            else:
                b=self.d[self.p]; self.p+=1
                if b==255 and self.p<len(self.d):
                    if self.d[self.p]==0: self.p+=1
                    else: b=255
            self.buf=(self.buf<<8)|b; self.n+=8
        self.n-=count
        out=(self.buf>>self.n)&((1<<count)-1)
        self.buf &= (1<<self.n)-1 if self.n else 0
        return out

class Huffman:
    def __init__(self, counts, symbols):
        self.lookup={}; code=0; ix=0
        for length, count in enumerate(counts,1):
            for _ in range(count): self.lookup[(length,code)]=symbols[ix]; code+=1; ix+=1
            code <<= 1
    def symbol(self, bits):
        code=0
        for n in range(1,17):
            code=(code<<1)|bits.get(1)
            if (n,code) in self.lookup: return self.lookup[(n,code)]
        raise ValueError('invalid Huffman code')

def decode_jpeg(data):
    if data[:2]!=b'\xff\xd8': raise ValueError('not jpeg')
    pos=2; quant={}; dc={}; ac={}; comps=[]; width=height=0
    while pos+4<=len(data):
        if data[pos]!=255: pos+=1; continue
        marker=data[pos+1]
        if marker in (0xd8,0xd9) or 0xd0<=marker<=0xd7: pos+=2; continue
        size=int.from_bytes(data[pos+2:pos+4],'big'); seg=data[pos+4:pos+2+size]
        if marker==0xdb:
            j=0
            while j<len(seg):
                spec=seg[j]; j+=1; tid=spec&15
                if spec>>4: quant[tid]=[int.from_bytes(seg[j+k*2:j+k*2+2],'big') for k in range(64)]; j+=128
                else: quant[tid]=list(seg[j:j+64]); j+=64
        elif marker==0xc0:
            height,width=struct.unpack('>HH',seg[1:5]); n=seg[5]
            for k in range(n):
                ident,hv,tq=seg[6+3*k:9+3*k]; comps.append({'id':ident,'h':hv>>4,'v':hv&15,'tq':tq})
        elif marker==0xc4:
            j=0
            while j<len(seg):
                spec=seg[j]; j+=1; counts=list(seg[j:j+16]); j+=16; syms=list(seg[j:j+sum(counts)]); j+=sum(counts)
                (dc if spec>>4==0 else ac)[spec&15]=Huffman(counts,syms)
        elif marker==0xda:
            n=seg[0]; scan=[]
            for k in range(n):
                ident,sel=seg[1+2*k:3+2*k]; scan.append((ident,sel>>4,sel&15))
            start=pos+2+size; break
        pos+=2+size
    else: raise ValueError('missing scan')
    hm=max(c['h'] for c in comps); vm=max(c['v'] for c in comps)
    mx=(width+8*hm-1)//(8*hm); my=(height+8*vm-1)//(8*vm)
    grids=[[[0]*(mx*c['h']) for _ in range(my*c['v'])] for c in comps]
    indexes={c['id']:i for i,c in enumerate(comps)}; previous=[0]*len(comps); bits=Bits(data,start)
    def extend(v,n): return v-(1<<n)+1 if n and v < (1<<(n-1)) else v
    for y in range(my):
        for x in range(mx):
            for ident,dt,at in scan:
                ci=indexes[ident]; c=comps[ci]
                for by in range(c['v']):
                    for bx in range(c['h']):
                        cat=dc[dt].symbol(bits); previous[ci]+=extend(bits.get(cat),cat)
                        grids[ci][y*c['v']+by][x*c['h']+bx]=previous[ci]*quant[c['tq']][0]
                        k=0
                        while k<63:
                            code=ac[at].symbol(bits)
                            if code==0: break
                            run=code>>4; n=code&15
                            if code==0xf0: k+=16
                            else:
                                if n: bits.get(n)
                                k+=run+1
    rows=my*vm; cols=mx*hm
    if len(comps)<3: raise ValueError('unexpected colour components')
    rgb=[]
    for r in range(rows):
        row=[]
        for col in range(cols):
            vals=[]
            for i,g in enumerate(grids):
                yy=min(r*len(g)//rows,len(g)-1); xx=min(col*len(g[0])//cols,len(g[0])-1)
                vals.append(g[yy][xx]/8+128)
            yy,cb,cr=vals[0],vals[1]-128,vals[2]-128
            row.append((max(0,min(255,yy+1.402*cr)),max(0,min(255,yy-.34414*cb-.71414*cr)),max(0,min(255,yy+1.772*cb))) )
        rgb.append(row)
    return rgb

def features(rgb):
    h=len(rgb); w=len(rgb[0]); flat=[p for row in rgb for p in row]
    def avg(a): return tuple(sum(p[i] for p in a)/len(a) for i in range(3))
    sky=avg([p for row in rgb[:max(1,int(h*.18))] for p in row]); mid=avg([p for row in rgb[int(h*.30):max(int(h*.65),int(h*.30)+1)] for p in row]); ground=avg([p for row in rgb[int(h*.72):] for p in row])
    lum=[.299*r+.587*g+.114*b for r,g,b in flat]; mu=sum(lum)/len(lum); ys=math.sqrt(sum((v-mu)**2 for v in lum)/len(lum))
    green=blue=white=gray=yellow=0
    for r,g,b in flat:
        green+=g>r+8 and g>b+5; blue+=b>r+20 and b>g+8
        white+=r>195 and g>195 and b>195 and max(r,g,b)-min(r,g,b)<25
        gray+=max(r,g,b)-min(r,g,b)<14 and 50<(r+g+b)/3<175
        yellow+=r>140 and g>105 and b<110 and r-b>50
    top=sum(.299*r+.587*g+.114*b for row in rgb[:h//3] for r,g,b in row)/(h//3*w)
    bot=sum(.299*r+.587*g+.114*b for row in rgb[2*h//3:] for r,g,b in row)/(h//3*w)
    left=sum(.299*r+.587*g+.114*b for row in rgb for r,g,b in row[:w//2])/(h*(w//2))
    right=sum(.299*r+.587*g+.114*b for row in rgb for r,g,b in row[w//2:])/(h*(w-w//2))
    return [blue/len(flat),sky[2]-sky[0],max(sky)-min(sky),mid[2],avg(flat)[2],mid[1]-mid[0],mid[1],gray/len(flat),ground[2],yellow/len(flat),white/len(flat),ground[1],ground[0],ground[0]-ground[1],left-right,ys]

def distance(a,b):
    return math.sqrt(sum((((a[i]-MEANS[i])/SDS[i])-((b[i]-MEANS[i])/SDS[i]))**2 for i in range(len(FEATS))))
def hav(a,b):
    p1,p2=math.radians(a[0]),math.radians(b[0]); dp=p2-p1; dl=math.radians(b[1]-a[1])
    z=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 12742*math.asin(min(1,math.sqrt(z)))
def median_point(points):
    lat=sum(p[0] for p in points)/len(points); lon=sum(p[1] for p in points)/len(points)
    for _ in range(50):
        nlat=nlon=den=0.0
        for la,lo in points:
            d=hav((lat,lon),(la,lo))
            if d<1e-9: return la,lo
            wt=1/d; nlat+=la*wt; nlon+=lo*wt; den+=wt
        la,lo=nlat/den,nlon/den
        if abs(la-lat)<1e-6 and abs(lo-lon)<1e-6: break
        lat,lon=la,lo
    return lat,lon
def locate(data):
    f=features(decode_jpeg(data)); ranked=sorted(ANCHORS,key=lambda a:distance(f,a[2]))
    return median_point([(a[0],a[1]) for a in ranked[:3]]+[CENTROID])
def request(url, body=None):
    req=urllib.request.Request(url,data=body,headers={'Content-Type':'application/json'} if body is not None else {})
    with urllib.request.urlopen(req,timeout=60) as response: return response.read()
def main():
    base=os.environ['MAKEFUN_ENV_URL'].rstrip('/')
    photos=json.loads(request(base+'/photos'))['photos']; answers=[]
    for p in photos:
        try: lat,lon=locate(request(base+p['url']))
        except Exception: lat,lon=CENTROID
        answers.append({'id':p['id'],'lat':round(lat,6),'lon':round(lon,6)})
    request(base+'/answers',json.dumps({'answers':answers},separators=(',',':')).encode())
if __name__=='__main__': main()
