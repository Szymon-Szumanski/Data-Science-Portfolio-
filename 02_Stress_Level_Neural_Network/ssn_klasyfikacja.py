import numpy as np
import pandas as pd
import csv, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

WYNIKI_DIR  = "results"
WYKRESY_DIR = os.path.join(WYNIKI_DIR, "charts")
os.makedirs(WYNIKI_DIR,  exist_ok=True)
os.makedirs(WYKRESY_DIR, exist_ok=True)

def wczytaj_dane(sciezka):
    return pd.read_csv(sciezka)


def przetworz_dane(df):
    df = df.copy()
    y = df["Stress Label"].apply(
        lambda x: 0 if x in ["Low Stress", "Moderate Stress"] else 1
    ).values.astype(int)
    kol_do_usuniecia = [
        "Stress Label", "Stress Value",
        df.columns[16], df.columns[17], df.columns[18], df.columns[19],
        df.columns[20], df.columns[21], df.columns[22], df.columns[23],
        df.columns[24], df.columns[25],
    ]
    df = df.drop(columns=kol_do_usuniecia)
    df["2. Gender"] = (df["2. Gender"] == "Female").astype(float)
    df["7. Did you receive a waiver or scholarship at your university?"] = (
        df["7. Did you receive a waiver or scholarship at your university?"] == "Yes"
    ).astype(float)

    wiek_mapa = {"Below 18": 0, "18-22": 1, "23-26": 2, "27-30": 3, "Above 30": 4}
    cgpa_mapa = {"Below 2.50": 0, "2.50 - 2.99": 1, "3.00 - 3.39": 2,
                 "3.40 - 3.79": 3, "3.80 - 4.00": 4, "Other": 2}
    rok_mapa  = {"First Year or Equivalent": 1, "Second Year or Equivalent": 2,
                 "Third Year or Equivalent": 3, "Fourth Year or Equivalent": 4, "Other": 2}
    anxiety_mapa = {"Minimal Anxiety": 0, "Mild Anxiety": 1,
                    "Moderate Anxiety": 2, "Severe Anxiety": 3}
    depr_mapa = {"No Depression": 0, "Minimal Depression": 1,
                 "Mild Depression": 2, "Moderate Depression": 3,
                 "Moderately Severe Depression": 4, "Severe Depression": 5}

    df["1. Age"]                = df["1. Age"].map(wiek_mapa)
    df["6. Current CGPA"]       = df["6. Current CGPA"].map(cgpa_mapa)
    df["5. Academic Year"]      = df["5. Academic Year"].map(rok_mapa)
    df["Anxiety Label"]         = df["Anxiety Label"].map(anxiety_mapa)
    df["Depression Label"]      = df["Depression Label"].map(depr_mapa)

    for kol, px in [("3. University", "Ucz"), ("4. Department", "Wyd")]:
        df = pd.concat([df, pd.get_dummies(df[kol], prefix=px)], axis=1)
        df = df.drop(kol, axis=1)

    X = df.values.astype(float)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        if m.any():
            X[m, j] = np.median(X[~m, j]) if (~m).any() else 0.0

    Xmin = X.min(0); Xr = X.max(0) - Xmin; Xr[Xr == 0] = 1.0
    return (X - Xmin) / Xr, y, Xmin, Xr

def podziel_dane(X, y, ut=0.2, z=42):
    np.random.seed(z); idx=np.random.permutation(len(y)); nt=int(len(y)*ut)
    return X[idx[nt:]], X[idx[:nt]], y[idx[nt:]], y[idx[:nt]]

def zbalansuj_dane_uczace(X_ucz, y_ucz):
    klasy, licznosc = np.unique(y_ucz, return_counts=True)

    if len(klasy) < 2 or abs(licznosc[0] - licznosc[1]) / max(licznosc) < 0.05:
        return X_ucz, y_ucz

    max_licznosc = max(licznosc)
    X_zbalansowane = []
    y_zbalansowane = []

    for klasa in klasy:
        indeksy_klasy = np.where(y_ucz == klasa)[0]
        wybrane_indeksy = np.random.choice(indeksy_klasy, size=max_licznosc, replace=True)
        X_zbalansowane.append(X_ucz[wybrane_indeksy])
        y_zbalansowane.append(y_ucz[wybrane_indeksy])

    X_ucz_bal = np.vstack(X_zbalansowane)
    y_ucz_bal = np.concatenate(y_zbalansowane)

    indeksy_shuf = np.arange(len(y_ucz_bal))
    np.random.shuffle(indeksy_shuf)

    return X_ucz_bal[indeksy_shuf], y_ucz_bal[indeksy_shuf]

def one_hot(y, n):
    oh=np.zeros((len(y),n)); oh[np.arange(len(y)),y]=1.0; return oh

class BatchNorm:

    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps=eps; self.momentum=momentum
        self.gamma=np.ones((1,dim)); self.beta=np.zeros((1,dim))
        self.run_mean=np.zeros((1,dim)); self.run_var=np.ones((1,dim))
        self.d_gamma=np.zeros_like(self.gamma); self.d_beta=np.zeros_like(self.beta)

    def naprzod(self, X, trening=True):
        if trening:
            self._mu  = X.mean(0,keepdims=True)
            self._var = X.var(0, keepdims=True)+self.eps
            self._Xn  = (X-self._mu)/np.sqrt(self._var)
            self.run_mean = self.momentum*self.run_mean+(1-self.momentum)*self._mu
            self.run_var  = self.momentum*self.run_var +(1-self.momentum)*self._var
        else:
            self._Xn = (X-self.run_mean)/np.sqrt(self.run_var+self.eps)
        return self.gamma*self._Xn+self.beta

    def wstecz(self, dY):
        m=dY.shape[0]
        dY = np.clip(dY, -10.0, 10.0)
        self.d_gamma=np.sum(dY*self._Xn,   axis=0,keepdims=True)
        self.d_beta =np.sum(dY,             axis=0,keepdims=True)
        dXn  = dY*self.gamma
        dvar = np.sum(dXn*self._Xn*(-.5)/self._var, axis=0,keepdims=True)
        dmu  = np.sum(dXn*(-1/np.sqrt(self._var)),  axis=0,keepdims=True)
        dX   = dXn/np.sqrt(self._var)+dvar*2*(self._Xn*np.sqrt(self._var))/m+dmu/m
        return np.clip(dX, -10.0, 10.0)

class SiecNeuronowa:

    def __init__(self, rozmiary_warstw, aktywacja="relu",
                 dropout_p=0.0, l2_lambda=0.0, uzywaj_bn=True, ziarno=None):
        if ziarno is not None: np.random.seed(ziarno)
        self.rozmiary  = rozmiary_warstw
        self.aktywacja = aktywacja
        self.dropout_p = dropout_p
        self.l2_lambda = l2_lambda
        self.uzywaj_bn = uzywaj_bn
        self.n_warstw  = len(rozmiary_warstw)-1
        self.W, self.b, self.bn = [], [], []
        for i in range(self.n_warstw):
            fi,fo = rozmiary_warstw[i], rozmiary_warstw[i+1]
            s = np.sqrt(2./fi) if aktywacja in ("relu","leaky_relu") else np.sqrt(2./(fi+fo))
            self.W.append(np.random.randn(fi,fo)*s)
            self.b.append(np.zeros((1,fo)))
            self.bn.append(BatchNorm(fo) if uzywaj_bn and i<self.n_warstw-1 else None)

    def _akt(self, Z):
        a=self.aktywacja
        if a=="sigmoid":    return 1/(1+np.exp(-np.clip(Z,-500,500)))
        if a=="tanh":       return np.tanh(Z)
        if a=="relu":       return np.maximum(0.,Z)
        if a=="leaky_relu": return np.where(Z>0,Z,.01*Z)
        return Z

    def _dakt(self, Z, A):
        a=self.aktywacja
        if a=="sigmoid":    return A*(1-A)
        if a=="tanh":       return 1-A**2
        if a=="relu":       return (Z>0).astype(float)
        if a=="leaky_relu": return np.where(Z>0,1.,.01)
        return np.ones_like(Z)

    @staticmethod
    def _softmax(Z):
        e=np.exp(Z-np.max(Z,1,keepdims=True)); return e/e.sum(1,keepdims=True)

    def propagacja_wprzod(self, X, trening=False):
        self._Z=[]; self._A=[X]; self._Abn=[]; self._mask=[]
        A=X
        for i in range(self.n_warstw-1):
            Z=A@self.W[i]+self.b[i]; self._Z.append(Z)
            A=self._akt(Z)
            if self.uzywaj_bn and self.bn[i]: A=self.bn[i].naprzod(A,trening)
            self._Abn.append(A.copy())
            if trening and self.dropout_p>0:
                m=(np.random.rand(*A.shape)>self.dropout_p).astype(float)
                A*=m/(1-self.dropout_p)
            else: m=np.ones_like(A)
            self._mask.append(m); self._A.append(A)
        Zo=A@self.W[-1]+self.b[-1]; self._Z.append(Zo)
        Ao=self._softmax(Zo);       self._A.append(Ao)
        return Ao

    def przewiduj(self, X):
        return np.argmax(self.propagacja_wprzod(X,False),axis=1)

    def prawdopodobienstwa(self, X):
        return self.propagacja_wprzod(X,False)

    def strata(self, Yp, Yo):
        ce=-np.mean(np.sum(Yo*np.log(Yp+1e-9),axis=1))
        if self.l2_lambda>0: ce+=self.l2_lambda*sum(np.sum(w**2) for w in self.W)
        return ce

    def dokladnosc(self, X, y):
        return float(np.mean(self.przewiduj(X)==y))

    def wsteczna_propagacja(self, Yo):
        m=Yo.shape[0]; gW=[None]*self.n_warstw; gb=[None]*self.n_warstw
        dA=self._A[-1]-Yo
        for i in range(self.n_warstw-1,-1,-1):
            gW[i]=(self._A[i].T@dA)/m
            gb[i]=np.mean(dA,axis=0,keepdims=True)
            if self.l2_lambda>0: gW[i]+=2*self.l2_lambda*self.W[i]
            if i>0:
                dA=(dA@self.W[i].T)*self._dakt(self._Z[i-1],self._Abn[i-1])
                dA=np.clip(dA, -10.0, 10.0)  # gradient clipping
                if self.uzywaj_bn and self.bn[i-1]: dA=self.bn[i-1].wstecz(dA)
                dA*=self._mask[i-1]
        return gW,gb

    def init_optymalizator(self, n):
        self._opt=n
        bns=[b for b in self.bn if b]
        if n in ("momentum","rmsprop"):
            self._vW=[np.zeros_like(w) for w in self.W]
            self._vb=[np.zeros_like(b) for b in self.b]
            self._vg=[np.zeros_like(b.gamma) for b in bns]
            self._vbt=[np.zeros_like(b.beta)  for b in bns]
        elif n=="adam":
            self._mW=[np.zeros_like(w) for w in self.W]
            self._mb=[np.zeros_like(b) for b in self.b]
            self._vW=[np.zeros_like(w) for w in self.W]
            self._vb=[np.zeros_like(b) for b in self.b]
            self._mg=[np.zeros_like(b.gamma) for b in bns]
            self._mbt=[np.zeros_like(b.beta)  for b in bns]
            self._vg=[np.zeros_like(b.gamma) for b in bns]
            self._vbt=[np.zeros_like(b.beta)  for b in bns]
            self._t=0

    def krok(self, gW, gb, lr):
        n=self._opt; bns=[b for b in self.bn if b]
        if n=="sgd":
            for i in range(self.n_warstw):
                self.W[i]-=lr*gW[i]; self.b[i]-=lr*gb[i]
            for b in bns: b.gamma-=lr*b.d_gamma; b.beta-=lr*b.d_beta
        elif n=="momentum":
            for i in range(self.n_warstw):
                self._vW[i]=.9*self._vW[i]+.1*gW[i]
                self._vb[i]=.9*self._vb[i]+.1*gb[i]
                self.W[i]-=lr*self._vW[i]; self.b[i]-=lr*self._vb[i]
            for j,b in enumerate(bns):
                self._vg[j]=.9*self._vg[j]+.1*b.d_gamma
                self._vbt[j]=.9*self._vbt[j]+.1*b.d_beta
                b.gamma-=lr*self._vg[j]; b.beta-=lr*self._vbt[j]
        elif n=="rmsprop":
            for i in range(self.n_warstw):
                self._vW[i]=.9*self._vW[i]+.1*gW[i]**2
                self._vb[i]=.9*self._vb[i]+.1*gb[i]**2
                self.W[i]-=lr*gW[i]/(np.sqrt(self._vW[i])+1e-8)
                self.b[i]-=lr*gb[i]/(np.sqrt(self._vb[i])+1e-8)
        elif n=="adam":
            b1,b2,eps=.9,.999,1e-8; self._t+=1
            for i in range(self.n_warstw):
                self._mW[i]=b1*self._mW[i]+(1-b1)*gW[i]
                self._mb[i]=b1*self._mb[i]+(1-b1)*gb[i]
                self._vW[i]=b2*self._vW[i]+(1-b2)*gW[i]**2
                self._vb[i]=b2*self._vb[i]+(1-b2)*gb[i]**2
                mwc=self._mW[i]/(1-b1**self._t); mbc=self._mb[i]/(1-b1**self._t)
                vwc=self._vW[i]/(1-b2**self._t); vbc=self._vb[i]/(1-b2**self._t)
                self.W[i]-=lr*mwc/(np.sqrt(vwc)+eps)
                self.b[i]-=lr*mbc/(np.sqrt(vbc)+eps)
            for j,b in enumerate(bns):
                self._mg[j]=b1*self._mg[j]+(1-b1)*b.d_gamma
                self._mbt[j]=b1*self._mbt[j]+(1-b1)*b.d_beta
                self._vg[j]=b2*self._vg[j]+(1-b2)*b.d_gamma**2
                self._vbt[j]=b2*self._vbt[j]+(1-b2)*b.d_beta**2
                mgc=self._mg[j]/(1-b1**self._t); mbtc=self._mbt[j]/(1-b1**self._t)
                vgc=self._vg[j]/(1-b2**self._t); vbtc=self._vbt[j]/(1-b2**self._t)
                b.gamma-=lr*mgc/(np.sqrt(vgc)+eps)
                b.beta -=lr*mbtc/(np.sqrt(vbtc)+eps)

class Ensemble:

    def __init__(self, n_sieci=10):
        self.n_sieci=n_sieci; self.sieci=[]

    def trenuj(self, X_ucz, y_ucz, X_test, y_test, rozmiary,
               aktywacja="relu", dropout_p=0.1, l2_lambda=0.001,
               uzywaj_bn=True, optymalizator="sgd", lr=0.05,
               epoki=300, rozmiar_batcha=64, harmonogram_lr=True,
               wczesne_zatrzymanie=True, cierpliwosc=40, verbose=True):
        self.sieci=[]
        for i in range(self.n_sieci):
            if verbose: print(f"    Network {i+1:2d}/{self.n_sieci} ...", end=" ", flush=True)
            s=SiecNeuronowa(rozmiary, aktywacja=aktywacja, dropout_p=dropout_p,
                            l2_lambda=l2_lambda, uzywaj_bn=uzywaj_bn, ziarno=i*97+13)
            ucz(s, X_ucz, y_ucz, X_test, y_test,
                optymalizator=optymalizator, lr=lr, epoki=epoki,
                rozmiar_batcha=rozmiar_batcha, harmonogram_lr=harmonogram_lr,
                wczesne_zatrzymanie=wczesne_zatrzymanie, cierpliwosc=cierpliwosc,
                verbose=False, co_ile=9999)
            self.sieci.append(s)
            if verbose:
                a=s.dokladnosc(X_test,y_test)
                f=f1_score_macro(y_test,s.przewiduj(X_test))
                print(f"acc={a:.4f}  F1={f:.4f}")
        return self

    def przewiduj_proba(self, X):
        return np.mean([s.prawdopodobienstwa(X) for s in self.sieci], axis=0)

    def przewiduj(self, X):
        return np.argmax(self.przewiduj_proba(X), axis=1)

    def dokladnosc(self, X, y):
        return float(np.mean(self.przewiduj(X)==y))

def macierz_pomylek(y_true, y_pred, n=2):
    M=np.zeros((n,n),dtype=int)
    for t,p in zip(y_true,y_pred): M[t,p]+=1
    return M

def f1_score_macro(y_true, y_pred, n_klas=2):
    f1s=[]
    for c in range(n_klas):
        tp=np.sum((y_true==c)&(y_pred==c))
        fp=np.sum((y_true!=c)&(y_pred==c))
        fn=np.sum((y_true==c)&(y_pred!=c))
        pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9)
        f1s.append(2*pr*rc/(pr+rc+1e-9))
    return float(np.mean(f1s))

class WczesneZatrzymanie:
    def __init__(self, cierpliwosc=40, delta=1e-4):
        self.cierpliwosc=cierpliwosc; self.delta=delta
        self._licznik=0; self._najlepsza=np.inf
    def sprawdz(self, s):
        if s<self._najlepsza-self.delta: self._najlepsza=s; self._licznik=0
        else: self._licznik+=1
        return self._licznik>=self.cierpliwosc

def ucz(siec, X_ucz, y_ucz, X_test, y_test,
        optymalizator="sgd", lr=0.05, epoki=300,
        rozmiar_batcha=64, harmonogram_lr=True,
        wczesne_zatrzymanie=True, cierpliwosc=40,
        verbose=True, co_ile=25):

    n_klas=siec.rozmiary[-1]; Y_oh=one_hot(y_ucz,n_klas); n=X_ucz.shape[0]
    hist={k:[] for k in ("epoka","train_loss","train_acc","test_loss","test_acc","f1_test")}
    siec.init_optymalizator(optymalizator)
    wz=WczesneZatrzymanie(cierpliwosc=cierpliwosc); lr_b=lr

    for epoka in range(1,epoki+1):
        if harmonogram_lr and epoka>1 and (epoka-1)%60==0: lr_b*=0.5
        idx=np.random.permutation(n)
        for s in range(0,n,rozmiar_batcha):
            ii=idx[s:s+rozmiar_batcha]
            siec.propagacja_wprzod(X_ucz[ii],trening=True)
            gW,gb=siec.wsteczna_propagacja(Y_oh[ii])
            siec.krok(gW,gb,lr_b)

        if epoka%co_ile==0 or epoka==1:
            Ypu=siec.propagacja_wprzod(X_ucz,False); Ypt=siec.propagacja_wprzod(X_test,False)
            tl=siec.strata(Ypu,Y_oh); ta=siec.dokladnosc(X_ucz,y_ucz)
            sl=siec.strata(Ypt,one_hot(y_test,n_klas)); sa=siec.dokladnosc(X_test,y_test)
            f1=f1_score_macro(y_test,siec.przewiduj(X_test))
            hist["epoka"].append(epoka); hist["train_loss"].append(round(tl,6))
            hist["train_acc"].append(round(ta,6)); hist["test_loss"].append(round(sl,6))
            hist["test_acc"].append(round(sa,6));  hist["f1_test"].append(round(f1,6))
            if verbose:
                print(f"    ep {epoka:4d}/{epoki}  train L={tl:.4f} A={ta:.4f}  "
                      f"test L={sl:.4f} A={sa:.4f}  F1={f1:.4f}  lr={lr_b:.5f}")

        if wczesne_zatrzymanie:
            Ypw=siec.propagacja_wprzod(X_test,False)
            if wz.sprawdz(siec.strata(Ypw,one_hot(y_test,n_klas))):
                if verbose: print(f"    >>> Early stopping at epoch {epoka}")
                break
    return hist

def experiment(X, y, cfg, n_powtorzen=5, verbose_run=False):
    input_dim=X.shape[1]; n_klas=int(y.max())+1
    rozmiary=[input_dim]+list(cfg.get("warstwy_ukryte",[128,64]))+[n_klas]
    wyniki=[]
    for run in range(n_powtorzen):
        X_ucz,X_test,y_ucz,y_test=podziel_dane(X,y,cfg.get("udzial_test",.2),run*37+13)
        X_ucz, y_ucz = zbalansuj_dane_uczace(X_ucz, y_ucz)
        siec=SiecNeuronowa(rozmiary,
            aktywacja=cfg.get("aktywacja","relu"),
            dropout_p=cfg.get("dropout",.1),
            l2_lambda=cfg.get("l2",.001),
            uzywaj_bn=cfg.get("bn",True),
            ziarno=run*97+5)
        hist=ucz(siec,X_ucz,y_ucz,X_test,y_test,
            optymalizator=cfg.get("optymalizator","sgd"),
            lr=cfg.get("lr",.05),
            epoki=cfg.get("epoki",300),
            rozmiar_batcha=cfg.get("rozmiar_batcha",64),
            harmonogram_lr=cfg.get("harmonogram_lr",True),
            wczesne_zatrzymanie=cfg.get("wczesne_zatrzymanie",True),
            cierpliwosc=cfg.get("cierpliwosc",40),
            verbose=verbose_run, co_ile=25)
        wyniki.append({"run":run+1,
            "train_acc_final":hist["train_acc"][-1],
            "test_acc_final": hist["test_acc"][-1],
            "train_loss_final":hist["train_loss"][-1],
            "test_loss_final": hist["test_loss"][-1],
            "f1_test_final":   hist["f1_test"][-1],
            "historia":        hist})
    at=[r["test_acc_final"] for r in wyniki]; au=[r["train_acc_final"] for r in wyniki]
    ft=[r["f1_test_final"]  for r in wyniki]
    return {"configuration":cfg,"repetitions":wyniki,
            "avg_train":   round(float(np.mean(au)),5),
            "avg_test":  round(float(np.mean(at)),5),
            "best_test":round(float(np.max(at)), 5),
            "std_test":      round(float(np.std(at)), 5),
            "avg_f1":    round(float(np.mean(ft)),5),
            "best_f1":  round(float(np.max(ft)), 5)}

def zapisz_csv(lista, sciezka):
    if not lista: return
    with open(sciezka,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=lista[0].keys()); w.writeheader(); w.writerows(lista)
    print(f"  [CSV] {sciezka}")

def wynik_do_wiersza(exp, pv, res):
    pows=res.get("repetitions",[])
    w={"experiment":exp,"parameter":pv,
       "avg_train":res["avg_train"],"avg_test":res["avg_test"],
       "best_test":res["best_test"],"std_test":res["std_test"],
       "avg_f1":res["avg_f1"],"best_f1":res["best_f1"]}
    for i,r in enumerate(pows,1):
        w[f"run{i}_train"]=r.get("train_acc_final","")
        w[f"run{i}_test"] =r.get("test_acc_final","")
        w[f"run{i}_f1"]   =r.get("f1_test_final","")
    return w

def zapisz_raport(lista, sciezka, meta):
    sep="="*72; lin="-"*72
    with open(sciezka,"w",encoding="utf-8") as f:
        f.write(sep+"\n  REPORT – SSN PROJECT (CLASSIFICATION – FINAL VERSION)\n")
        f.write(f"  Date : {meta['data']}\n  Dataset: {meta['zbior']}\n")
        f.write(f"  Goal : {meta['cel']}\n")
        f.write(f"  Configuration: SGD + BatchNorm + CrossEntropy + Ensemble(10)\n"+sep+"\n")
        akt=None
        for w in lista:
            if w["experiment"]!=akt:
                akt=w["experiment"]
                f.write(f"\n{lin}\n  EXPERIMENT: {akt}\n{lin}\n")
                f.write(f"  {'Parameter':28s} {'Train':>7} {'Test_avg':>8} "
                        f"{'Test_best':>10} {'Std':>7} {'F1_avg':>7} {'F1_best':>8}\n")
                f.write(f"  {'─'*28} {'─'*7} {'─'*8} {'─'*10} {'─'*7} {'─'*7} {'─'*8}\n")
            f.write(f"  {str(w['parameter']):28s} {w['avg_train']:7.4f} "
                    f"{w['avg_test']:8.4f} {w['best_test']:10.4f} "
                    f"{w['std_test']:7.4f} {w['avg_f1']:7.4f} {w['best_f1']:8.4f}\n")
        f.write(f"\n{sep}\n")
    print(f"  [TXT] {sciezka}")


KOLORY=["#2196F3","#F44336","#4CAF50","#FF9800","#9C27B0","#00BCD4","#E91E63","#8BC34A"]

def wykres_macierzy_pomylek(y_true, y_pred, tytul, sciezka):
    n=int(max(y_true.max(),y_pred.max()))+1
    M=macierz_pomylek(y_true,y_pred,n)
    etyk=["Low/Moderate","High"] if n==2 else [str(i) for i in range(n)]
    fig,ax=plt.subplots(figsize=(5,4))
    im=ax.imshow(M,cmap="Blues")
    ax.set_xticks(list(range(n))); ax.set_yticks(list(range(n)))
    ax.set_xticklabels(etyk); ax.set_yticklabels(etyk)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(tytul)
    for i in range(n):
        for j in range(n):
            ax.text(j,i,str(M[i,j]),ha="center",va="center",fontsize=13,fontweight="bold",
                    color="white" if M[i,j]>M.max()*.5 else "black")
    plt.colorbar(im,ax=ax); plt.tight_layout()
    plt.savefig(sciezka,dpi=120,bbox_inches="tight"); plt.close(fig)
    print(f"  [PNG] {sciezka}")

def wykres_exp(wyniki_exp, tytul, param_label, png):
    n=len(wyniki_exp); etyk=[str(r["param_val"]) for r in wyniki_exp]
    fig=plt.figure(figsize=(16,5))
    fig.suptitle(f"Experiment: {tytul}",fontsize=14,fontweight="bold")
    gs=gridspec.GridSpec(1,3,figure=fig,wspace=.35)
    ax1=fig.add_subplot(gs[0]); x=np.arange(n); w=.35
    u_v=[r["res"]["avg_train"]    for r in wyniki_exp]
    t_v=[r["res"]["avg_test"]   for r in wyniki_exp]
    b_v=[r["res"]["best_test"] for r in wyniki_exp]
    s_v=[r["res"]["std_test"]       for r in wyniki_exp]
    b1=ax1.bar(x-w/2,u_v,w,label="Train (avg.)", color="#2196F3",alpha=.85)
    b2=ax1.bar(x+w/2,t_v,w,label="Test (avg.)",color="#F44336",alpha=.85,
               yerr=s_v,capsize=4,error_kw={"elinewidth":1.5})
    ax1.scatter(x+w/2,b_v,color="#FF9800",zorder=5,s=55,marker="D",label="Test (best)")
    for rect,v in zip(b1,u_v):
        ax1.text(rect.get_x()+rect.get_width()/2,rect.get_height()+.008,
                 f"{v:.3f}",ha="center",va="bottom",fontsize=7.5)
    for rect,v in zip(b2,t_v):
        ax1.text(rect.get_x()+rect.get_width()/2,rect.get_height()+.008,
                 f"{v:.3f}",ha="center",va="bottom",fontsize=7.5)
    ax1.set_xticks(x); ax1.set_xticklabels(etyk,rotation=20,ha="right",fontsize=9)
    ax1.set_ylabel("Accuracy"); ax1.set_title("Accuracy")
    ax1.set_ylim(0,1.1); ax1.legend(fontsize=8); ax1.grid(axis="y",alpha=.3)
    ax2=fig.add_subplot(gs[1])
    f1_v=[r["res"]["avg_f1"]   for r in wyniki_exp]
    f1b =[r["res"]["best_f1"] for r in wyniki_exp]
    bars=ax2.bar(x,f1_v,.6,color="#4CAF50",alpha=.85,label="F1 macro (avg.)")
    ax2.scatter(x,f1b,color="#FF9800",zorder=5,s=55,marker="D",label="F1 (best)")
    for i,v in enumerate(f1_v):
        ax2.text(i,v+.008,f"{v:.3f}",ha="center",va="bottom",fontsize=7.5)
    ax2.set_xticks(x); ax2.set_xticklabels(etyk,rotation=20,ha="right",fontsize=9)
    ax2.set_ylabel("F1-score (macro)"); ax2.set_title("F1-score")
    ax2.set_ylim(0,1.1); ax2.legend(fontsize=8); ax2.grid(axis="y",alpha=.3)
    ax3=fig.add_subplot(gs[2])
    for ir,r in enumerate(wyniki_exp):
        nb=max(r["res"]["repetitions"],key=lambda x:x["test_acc_final"])
        h=nb["historia"]; kol=KOLORY[ir%len(KOLORY)]
        ax3.plot(h["epoka"],h["test_acc"],color=kol,lw=2.,label=str(r["param_val"]))
        ax3.plot(h["epoka"],h["train_acc"],color=kol,lw=.9,linestyle="--",alpha=.45)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Accuracy")
    ax3.set_title("Learning curves\n(─ test,  ╌ train)")
    ax3.legend(title=param_label,fontsize=8); ax3.grid(alpha=.3); ax3.set_ylim(0,1.)
    plt.subplots_adjust(wspace=.35,top=.87,bottom=.16,left=.06,right=.97)
    plt.savefig(png,dpi=120,bbox_inches="tight"); plt.close(fig)
    print(f"  [PNG] {png}")

def wykres_podsumowanie(lista_csv, png):
    from collections import defaultdict
    grupy=defaultdict(list)
    for w in lista_csv: grupy[w["experiment"]].append(w)
    ne=len(grupy); nc=min(ne,5); nr=(ne+nc-1)//nc
    fig,axes=plt.subplots(nr,nc,figsize=(5*nc,4.5*nr))
    axes=np.array(axes).flatten()
    fig.suptitle("Summary – Test accuracy and F1 across all experiments",
                 fontsize=12,fontweight="bold")
    for i,(en,wiersze) in enumerate(grupy.items()):
        ax=axes[i]; params=[str(w["parameter"]) for w in wiersze]
        tv=[w["avg_test"] for w in wiersze]; fv=[w["avg_f1"] for w in wiersze]
        x=np.arange(len(params))
        ax.barh(x-.2,tv,.35,color="#2196F3",alpha=.8,label="Acc (avg.)")
        ax.barh(x+.2,fv,.35,color="#4CAF50",alpha=.8,label="F1 (avg.)")
        ax.set_yticks(x); ax.set_yticklabels(params,fontsize=8)
        ax.set_xlabel("Value",fontsize=8)
        ax.set_title(en.split("_",1)[-1],fontsize=9,fontweight="bold")
        ax.set_xlim(0,1.); ax.grid(axis="x",alpha=.3)
    for j in range(i+1,len(axes)): axes[j].set_visible(False)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=2,fontsize=9,bbox_to_anchor=(.5,-.02))
    plt.subplots_adjust(hspace=.5,wspace=.4,top=.90,bottom=.10,left=.06,right=.97)
    plt.savefig(png,dpi=120,bbox_inches="tight"); plt.close(fig)
    print(f"  [PNG] {png}")

def wykres_porownanie_ensemble(wyniki_single, wyniki_ens, png):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Single network vs Ensemble (10 networks)", fontsize=13, fontweight="bold")

    etyk = ["Low/Medium", "High"]
    for ax, (y_true, y_pred, tytul, acc, f1) in zip(axes, wyniki_ens):
        n = 2
        M = macierz_pomylek(y_true, y_pred, n)
        im = ax.imshow(M, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(etyk); ax.set_yticklabels(etyk)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{tytul}\nAcc={acc:.3f}  F1={f1:.3f}")
        for i in range(n):
            for j in range(n):
                ax.text(j,i,str(M[i,j]),ha="center",va="center",fontsize=12,
                        fontweight="bold",
                        color="white" if M[i,j]>M.max()*.5 else "black")
    plt.tight_layout()
    plt.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  [PNG] {png}")

def sep(c="=",n=68): print(c*n)
def h_exp(nr,t): print(); sep("-"); print(f"  EXP {nr}: {t}"); sep("-")
def drukuj(p,r):
    print(f"    [{str(p):25s}]  train={r['avg_train']:.4f}  "
          f"test={r['avg_test']:.4f}  best={r['best_test']:.4f}  "
          f"F1={r['avg_f1']:.4f}  std={r['std_test']:.4f}")

if __name__ == "__main__":

    START=time.time(); sep()
    print("  SSN PROJECT – STRESS CLASSIFICATION (Bangladesh, real survey data)")
    print("  Configuration: SGD + BatchNorm + CrossEntropy + Ensemble")
    print(f"  Start: {datetime.now():%Y-%m-%d %H:%M:%S}"); sep()

    SCIEZKA="Raw_Data.csv"
    df=wczytaj_dane(SCIEZKA)
    X,y,X_min,X_zakres=przetworz_dane(df)
    print(f"\n  Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"  Classes: Low/Moderate={np.sum(y==0)}  High={np.sum(y==1)}")

    N_POWT=5
    BASE=dict(
        warstwy_ukryte=[128,64], aktywacja="relu", optymalizator="sgd",
        lr=0.05, epoki=300, rozmiar_batcha=64, udzial_test=0.2,
        dropout=0.1, l2=0.001, bn=True,
        harmonogram_lr=True, wczesne_zatrzymanie=True, cierpliwosc=40,
    )

    wszystkie_csv=[]

    def uruchom_exp(nr, nazwa, param_label, configs):
        h_exp(nr, nazwa); wyniki_exp=[]
        for pv,cfg in configs:
            res=experiment(X,y,cfg,N_POWT,verbose_run=False)
            drukuj(pv,res)
            wszystkie_csv.append(wynik_do_wiersza(f"{nr:02d}_{nazwa}",str(pv),res))
            wyniki_exp.append({"param_val":pv,"res":res})
        wykres_exp(wyniki_exp,nazwa,param_label,
                   os.path.join(WYKRESY_DIR,f"exp{nr:02d}_{nazwa}.png"))

    uruchom_exp(1,"Number_of_layers","Architecture",[
        ([32],             {**BASE,"warstwy_ukryte":[32]}),
        ([64,32],          {**BASE,"warstwy_ukryte":[64,32]}),
        ([128,64],         {**BASE,"warstwy_ukryte":[128,64]}),
        ([128,64,32],      {**BASE,"warstwy_ukryte":[128,64,32]}),
        ([256,128,64,32],  {**BASE,"warstwy_ukryte":[256,128,64,32]}),
    ])
    uruchom_exp(2,"Number_of_neurons","Neurons",[
        (16, {**BASE,"warstwy_ukryte":[16]}),
        (32, {**BASE,"warstwy_ukryte":[32]}),
        (64, {**BASE,"warstwy_ukryte":[64]}),
        (128,{**BASE,"warstwy_ukryte":[128]}),
        (256,{**BASE,"warstwy_ukryte":[256]}),
    ])
    uruchom_exp(3,"Optimizer","Method",[
        ("sgd",     {**BASE,"optymalizator":"sgd",     "lr":0.05}),
        ("momentum",{**BASE,"optymalizator":"momentum","lr":0.01}),
        ("rmsprop", {**BASE,"optymalizator":"rmsprop", "lr":0.001}),
        ("adam",    {**BASE,"optymalizator":"adam",    "lr":0.001}),
    ])
    uruchom_exp(4,"Activation","Function",[
        ("relu",      {**BASE,"aktywacja":"relu"}),
        ("leaky_relu",{**BASE,"aktywacja":"leaky_relu"}),
        ("tanh",      {**BASE,"aktywacja":"tanh"}),
        ("sigmoid",   {**BASE,"aktywacja":"sigmoid"}),
    ])
    uruchom_exp(5,"Dataset_split","Test %",[
        ("10%",{**BASE,"udzial_test":0.10}),
        ("20%",{**BASE,"udzial_test":0.20}),
        ("30%",{**BASE,"udzial_test":0.30}),
        ("40%",{**BASE,"udzial_test":0.40}),
    ])
    uruchom_exp(6,"Batch_size","Batch size",[
        (16, {**BASE,"rozmiar_batcha":16}),
        (32, {**BASE,"rozmiar_batcha":32}),
        (64, {**BASE,"rozmiar_batcha":64}),
        (128,{**BASE,"rozmiar_batcha":128}),
        (256,{**BASE,"rozmiar_batcha":256}),
    ])
    uruchom_exp(7,"Learning_rate","lr",[
        (0.1,  {**BASE,"lr":0.1,  "harmonogram_lr":False}),
        (0.05, {**BASE,"lr":0.05, "harmonogram_lr":False}),
        (0.01, {**BASE,"lr":0.01, "harmonogram_lr":False}),
        (0.001,{**BASE,"lr":0.001,"harmonogram_lr":False}),
    ])
    uruchom_exp(8,"L2_regularization","lambda",[
        (0.0,   {**BASE,"l2":0.0}),
        (0.0001,{**BASE,"l2":0.0001}),
        (0.001, {**BASE,"l2":0.001}),
        (0.01,  {**BASE,"l2":0.01}),
        (0.1,   {**BASE,"l2":0.1}),
    ])
    uruchom_exp(9,"Dropout","p",[
        (0.0, {**BASE,"dropout":0.0}),
        (0.1, {**BASE,"dropout":0.1}),
        (0.2, {**BASE,"dropout":0.2}),
        (0.3, {**BASE,"dropout":0.3}),
        (0.5, {**BASE,"dropout":0.5}),
    ])
    uruchom_exp(10,"Number_of_epochs","Epochs",[
        (50, {**BASE,"epoki":50, "wczesne_zatrzymanie":False}),
        (100,{**BASE,"epoki":100,"wczesne_zatrzymanie":False}),
        (200,{**BASE,"epoki":200,"wczesne_zatrzymanie":False}),
        (350,{**BASE,"epoki":350,"wczesne_zatrzymanie":False}),
    ])
    uruchom_exp(11,"BatchNorm","BN",[
        ("without_BN",{**BASE,"bn":False}),
        ("with_BN",   {**BASE,"bn":True}),
    ])

    print(); sep()
    print("  COMPARISON: Single network  vs  Ensemble (10 networks)"); sep()

    X_ucz,X_test,y_ucz,y_test=podziel_dane(X,y,0.2,42)
    rozmiary=[X.shape[1],128,64,2]


    print("\n  [A] Single network:")
    siec_s=SiecNeuronowa(rozmiary,aktywacja="relu",dropout_p=0.1,
                         l2_lambda=0.001,uzywaj_bn=True,ziarno=42)
    ucz(siec_s,X_ucz,y_ucz,X_test,y_test,
        optymalizator="sgd",lr=0.05,epoki=300,rozmiar_batcha=64,
        harmonogram_lr=True,wczesne_zatrzymanie=True,cierpliwosc=40,
        verbose=True,co_ile=50)
    pred_s=siec_s.przewiduj(X_test)
    acc_s=siec_s.dokladnosc(X_test,y_test)
    f1_s=f1_score_macro(y_test,pred_s)
    print(f"\n  Single      →  Accuracy={acc_s:.4f}  F1={f1_s:.4f}")

    print("\n  [B] Ensemble (10 networks):")
    ens=Ensemble(n_sieci=10)
    ens.trenuj(X_ucz,y_ucz,X_test,y_test,rozmiary,
               aktywacja="relu",dropout_p=0.1,l2_lambda=0.001,uzywaj_bn=True,
               optymalizator="sgd",lr=0.05,epoki=300,rozmiar_batcha=64,
               harmonogram_lr=True,wczesne_zatrzymanie=True,cierpliwosc=40,
               verbose=True)
    pred_e=ens.przewiduj(X_test)
    acc_e=ens.dokladnosc(X_test,y_test)
    f1_e=f1_score_macro(y_test,pred_e)
    print(f"\n  Ensemble    →  Accuracy={acc_e:.4f}  F1={f1_e:.4f}")
    print(f"  Improvement →  Acc +{(acc_e-acc_s)*100:+.2f}%  F1 +{(f1_e-f1_s)*100:+.2f}%")

    wykres_macierzy_pomylek(y_test,pred_s,
        f"Single network (SGD+BN)\nAcc={acc_s:.3f}  F1={f1_s:.3f}",
        os.path.join(WYNIKI_DIR,"confusion_matrix_single.png"))
    wykres_macierzy_pomylek(y_test,pred_e,
        f"Ensemble – 10 networks (SGD+BN)\nAcc={acc_e:.3f}  F1={f1_e:.3f}",
        os.path.join(WYNIKI_DIR,"confusion_matrix_ensemble.png"))

    wykres_porownanie_ensemble(None,[
        (y_test,pred_s,"Single network",acc_s,f1_s),
        (y_test,pred_e,"Ensemble (10 networks)",acc_e,f1_e),
    ],os.path.join(WYNIKI_DIR,"ensemble_comparison.png"))

    print(); sep(); print("  SAVING RESULTS"); sep()
    zapisz_csv(wszystkie_csv,os.path.join(WYNIKI_DIR,"experiment_results.csv"))
    zapisz_raport(wszystkie_csv,os.path.join(WYNIKI_DIR,"report.txt"),
                  meta={"data":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "zbior":SCIEZKA,
                        "cel":"Stress Label -> Low/Moderate Stress(0) / High Perceived Stress(1)"})
    wykres_podsumowanie(wszystkie_csv,os.path.join(WYNIKI_DIR,"summary_all.png"))

    czas=(time.time()-START)/60; sep()
    print(f"  Total time     : {czas:.1f} min")
    print(f"  Directory      : {os.path.abspath(WYNIKI_DIR)}/")
    print(f"    experiment_results.csv")
    print(f"    report.txt")
    print(f"    confusion_matrix_single.png")
    print(f"    confusion_matrix_ensemble.png")
    print(f"    ensemble_comparison.png")
    print(f"    summary_all.png")
    print(f"    wykresy/  (exp01..exp11)")
    sep()