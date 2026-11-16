#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <string>
#include <omp.h>
using namespace std;

const double G=1.0, EPS2=1e-3, THETA=0.5;
const int LEAF_SIZE=8;
struct Particle {
    double x,y,z,vx,vy,vz,fx,fy,fz,mass;
    uint64_t morton_key;
    int work_cost,id;
};
struct Node {
    int lo,hi;
    double size,mass=0,cx=0,cy=0,cz=0;
    int child[8];
    Node(int l,int h,double s):lo(l),hi(h),size(s) {
        fill(child,child+8,-1);
    }
};
uint64_t expand_bits(uint64_t v) {
    v=(v|(v<<32))&0x001F00000000FFFFULL;
    v=(v|(v<<16))&0x001F0000FF0000FFULL;
    v=(v|(v<<8))&0x100F00F00F00F00FULL;
    v=(v|(v<<4))&0x10C30C30C30C30C3ULL;
    return (v|(v<<2))&0x1249249249249249ULL;
}

double morton_sort(vector<Particle>& p) {
    double xmin=p[0].x,ymin=p[0].y,zmin=p[0].z;
    double xmax=xmin,ymax=ymin,zmax=zmin;
    for(const auto& a:p) {
        xmin=min(xmin,a.x); xmax=max(xmax,a.x);
        ymin=min(ymin,a.y); ymax=max(ymax,a.y);
        zmin=min(zmin,a.z); zmax=max(zmax,a.z);
    }
    double side=max(1e-9,max(xmax-xmin,max(ymax-ymin,zmax-zmin)))*1.000001;
    auto grid=[&](double x,double low)->uint64_t {
        return static_cast<uint64_t>(max(0.0,min(2097151.0,(x-low)/side*2097152.0)));
    };

    #pragma omp parallel for
    for(int i=0;i<(int)p.size();++i) {
        p[i].morton_key=expand_bits(grid(p[i].x,xmin)) |
            (expand_bits(grid(p[i].y,ymin))<<1) |
            (expand_bits(grid(p[i].z,zmin))<<2);
    }
    sort(p.begin(),p.end(),[](const Particle& a,const Particle& b){
        return a.morton_key<b.morton_key;
    });
    return side;
}

int build_tree(const vector<Particle>& p,vector<Node>& tree,
               int lo,int hi,int level,double size) {
    int k=(int)tree.size();
    tree.emplace_back(lo,hi,size);
    if(hi-lo<=LEAF_SIZE || level<0) {
        for(int i=lo;i<hi;++i) {
            tree[k].mass+=p[i].mass;
            tree[k].cx+=p[i].mass*p[i].x;
            tree[k].cy+=p[i].mass*p[i].y;
            tree[k].cz+=p[i].mass*p[i].z;
        }
    } else {
        int i=lo;
        while(i<hi) {
            int oct=(p[i].morton_key>>(3*level))&7ULL;
            int j=i+1;
            while(j<hi && ((p[j].morton_key>>(3*level))&7ULL)==(uint64_t)oct) ++j;
            int c=build_tree(p,tree,i,j,level-1,size/2);
            tree[k].child[oct]=c;
            tree[k].mass+=tree[c].mass;
            tree[k].cx+=tree[c].mass*tree[c].cx;
            tree[k].cy+=tree[c].mass*tree[c].cy;
            tree[k].cz+=tree[c].mass*tree[c].cz;
            i=j;
        }
    }
    tree[k].cx/=tree[k].mass;
    tree[k].cy/=tree[k].mass;
    tree[k].cz/=tree[k].mass;
    return k;
}
void add_force(const Particle& a,double x,double y,double z,double mass,
               double& fx,double& fy,double& fz) {
    double dx=x-a.x,dy=y-a.y,dz=z-a.z;
    double r2=dx*dx+dy*dy+dz*dz+EPS2;
    double f=G*a.mass*mass/(r2*sqrt(r2));
    fx+=f*dx; fy+=f*dy; fz+=f*dz;
}
void tree_force(const vector<Particle>& p,const vector<Node>& tree,
                int k,int i,double& fx,double& fy,double& fz,int& cost) {
    const Node& n=tree[k];
    const Particle& a=p[i];
    ++cost; // count node visits
    bool leaf=true;
    for(int c:n.child) if(c!=-1) {leaf=false;break;}
    if(leaf) {
        for(int j=n.lo;j<n.hi;++j) if(j!=i) {
            add_force(a,p[j].x,p[j].y,p[j].z,p[j].mass,fx,fy,fz);
            ++cost;
        }
        return;
    }
    double dx=n.cx-a.x,dy=n.cy-a.y,dz=n.cz-a.z;
    double d2=dx*dx+dy*dy+dz*dz;
    bool contains=(i>=n.lo && i<n.hi);
    if(!contains && n.size*n.size<THETA*THETA*d2) {
        add_force(a,n.cx,n.cy,n.cz,n.mass,fx,fy,fz);
        ++cost;
        return;
    }
    for(int c:n.child) if(c!=-1) tree_force(p,tree,c,i,fx,fy,fz,cost);
}
vector<int> make_costzones(const vector<Particle>& p,int zones,bool adaptive) {
    int n=(int)p.size();
    vector<int> bounds(zones+1,0);
    bounds[zones]=n;
    if(!adaptive) {
        for(int z=1;z<zones;++z) bounds[z]=(long long)n*z/zones;
        return bounds;
    }
    long long total=0,running=0;
    for(const auto& a:p) total+=a.work_cost;
    int pos=0;
    for(int z=1;z<zones;++z) {
        long long target=total*z/zones;
        while(pos<n && running<target) running+=p[pos++].work_cost;
        bounds[z]=pos;
    }
    return bounds;
}

void compute_forces(vector<Particle>& p,int threads,int mode,bool report) {
    int n=(int)p.size();
    vector<Node> tree;
    if(mode!=0) {
        double side=morton_sort(p);
        build_tree(p,tree,0,n,20,side);
    }
    auto bounds=make_costzones(p,threads,mode==2);
    vector<long long> loads(threads,0);

    #pragma omp parallel for schedule(static,1) num_threads(threads)
    for(int z=0;z<threads;++z) {
        for(int i=bounds[z];i<bounds[z+1];++i) {
            double fx=0,fy=0,fz=0;
            int cost=0;
            if(mode==0) {
                for(int j=0;j<n;++j) if(i!=j) {
                    add_force(p[i],p[j].x,p[j].y,p[j].z,p[j].mass,fx,fy,fz);
                    ++cost;
                }
            } else tree_force(p,tree,0,i,fx,fy,fz,cost);
            p[i].fx=fx; p[i].fy=fy; p[i].fz=fz;
            p[i].work_cost=cost;
            loads[z]+=cost;
        }
    }
    if(report) {
        long long sum=0,peak=0;
        for(auto w:loads) {sum+=w;peak=max(peak,w);}
        cout<<"  Zone work max/mean: "<<(double)peak/(double(sum)/threads)<<'\n';
    }
}
void half_kick_and_drift(vector<Particle>& p,double dt) {
    #pragma omp parallel for
    for(int i=0;i<(int)p.size();++i) {
        auto& a=p[i];
        a.vx+=a.fx/a.mass*dt/2; a.vy+=a.fy/a.mass*dt/2; a.vz+=a.fz/a.mass*dt/2;
        a.x+=a.vx*dt; a.y+=a.vy*dt; a.z+=a.vz*dt;
    }
}
void final_half_kick(vector<Particle>& p,double dt) {
    #pragma omp parallel for
    for(int i=0;i<(int)p.size();++i) {
        auto& a=p[i];
        a.vx+=a.fx/a.mass*dt/2; a.vy+=a.fy/a.mass*dt/2; a.vz+=a.fz/a.mass*dt/2;
    }
}
double force_error(const vector<Particle>& exact,const vector<Particle>& approx) {
    double num=0,den=0;
    for(const auto& a:approx) {
        const auto& e=exact[a.id];
        num+=pow(a.fx-e.fx,2)+pow(a.fy-e.fy,2)+pow(a.fz-e.fz,2);
        den+=e.fx*e.fx+e.fy*e.fy+e.fz*e.fz;
    }
    return den>0?100*sqrt(num/den):0;
}
int main(int argc,char** argv) {
    int N=argc>1?atoi(argv[1]):10000;
    int threads=argc>2?atoi(argv[2]):min(8,omp_get_max_threads());
    if(N<2 || threads<1) return 1;
    threads=min(N,threads);
    const int STEPS=5;
    const double dt=0.01;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    srand(42);
    vector<Particle> initial(N);
    for(int i=0;i<N;++i) {
        auto& a=initial[i];
        double r=(double)rand()/RAND_MAX*50;
        a.x=r*(2.0*rand()/RAND_MAX-1);
        a.y=r*(2.0*rand()/RAND_MAX-1);
        a.z=r*(2.0*rand()/RAND_MAX-1);
        a.mass=1; a.work_cost=N-1; a.id=i;
    }
    cout<<"N="<<N<<", threads="<<threads<<", theta="<<THETA<<'\n';
    auto exact=initial, approx=initial;
    compute_forces(exact,threads,0,false);
    compute_forces(approx,threads,1,false);
    double error=force_error(exact,approx);
    cout<<"Initial relative L2 force error: "<<error<<"%\n";
    const char* names[]={"Direct","Barnes-Hut static","Barnes-Hut costzones"};
    double direct_time=0;
    const string filename="benchmark_barnes_hut.csv";
    bool exists=ifstream(filename).good();
    ofstream csv(filename,ios::app);
    if(!exists) csv<<"N,Threads,Mode,Theta,Seconds,Initial_Force_Error_Percent\n";
    for(int mode=0;mode<3;++mode) {
        auto p=initial;
        cout<<names[mode]<<'\n';
        double start=omp_get_wtime();
        compute_forces(p,threads,mode,false);
        for(int step=0;step<STEPS;++step) {
            half_kick_and_drift(p,dt);
            compute_forces(p,threads,mode,false);
            final_half_kick(p,dt);
        }
        double seconds=omp_get_wtime()-start;
        if(mode==0) direct_time=seconds;
        cout<<"  Time: "<<seconds<<" s, speedup vs direct: "<<direct_time/seconds<<"x\n";
        csv<<N<<','<<threads<<','<<names[mode]<<','<<THETA<<','<<seconds<<','<<(mode==0?0:error)<<'\n';
        compute_forces(p,threads,mode,true);
    }
}
