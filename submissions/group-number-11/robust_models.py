"""Independent robustness candidates: formatting invariance and content-light stylometry."""
import re,unicodedata
from collections import Counter
import numpy as np
import scipy.sparse as sp
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

SEED=42
FUNCTION_WORDS=sorted(ENGLISH_STOP_WORDS)
HOMO=str.maketrans({'а':'a','е':'e','о':'o','р':'p','с':'c','у':'y','х':'x','і':'i',
                    '“':'"','”':'"','‘':"'",'’':"'",'–':'-','—':'-'})

def normalize(text):
    text=unicodedata.normalize('NFKC',str(text)).translate(HOMO)
    text=''.join(c for c in text if unicodedata.category(c)!='Cf')
    return ' '.join(text.split())


def style_features(texts):
    rows=[]
    for text in texts:
        # Deliberately remove line-wrap and typography shortcuts before describing style.
        t=normalize(text);words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",t);lower=[w.lower() for w in words]
        counts=Counter(lower);n=max(1,len(words));chars=max(1,len(t));lengths=np.array([len(w) for w in words] or [0])
        sentences=[s for s in re.split(r'[.!?]+',t) if s.strip()]
        sl=np.array([len(s.split()) for s in sentences] or [0]);token_p=np.array(list(counts.values()))/n
        normalized_sentences=[re.sub(r'\W+',' ',s.lower()).strip() for s in sentences]
        row=[np.log1p(n),np.log1p(chars),len(counts)/n,sum(v==1 for v in counts.values())/n,
             -np.sum(token_p*np.log(token_p+1e-12)),lengths.mean(),lengths.std(),
             *np.quantile(lengths,[.1,.25,.5,.75,.9]),len(sentences)/n,sl.mean(),sl.std(),
             *np.quantile(sl,[.1,.5,.9]),max(sl)/n,
             sum(w[0].isupper() for w in words)/n,sum(w.isupper() and len(w)>1 for w in words)/n,
             sum("'" in w for w in words)/n,sum(c.isdigit() for c in t)/chars,
             sum(ord(c)>127 for c in t)/chars,
             1-len(set(normalized_sentences))/max(1,len(normalized_sentences)),
             max(Counter(normalized_sentences).values(),default=0)/max(1,len(sentences)),
             max(counts.values(),default=0)/n]
        row += [np.mean(lengths==i) for i in range(1,21)]
        row += [t.count(c)/chars for c in '.,;:!?-()[]{}"/\\%$&*+=<>']
        row += [counts[w]/n for w in FUNCTION_WORDS]
        # Character frequencies and local word-length transitions are topic-light.
        row += [t.lower().count(c)/chars for c in 'abcdefghijklmnopqrstuvwxyz']
        bins=np.minimum(lengths,12)
        row += [float(np.mean((bins[:-1]==i)&(bins[1:]==j))) if len(bins)>1 else 0.
                for i in (1,2,3,4,5,6,7,8,9,10,11,12) for j in (1,2,3,4,5,6,7,8,9,10,11,12)]
        rows.append(row)
    return np.asarray(rows,dtype=np.float32)


class RobustModel:
    def __init__(self,kind):self.kind=kind

    def fit(self,texts,y):
        texts=list(texts);y=np.asarray(y)
        if self.kind=='style':
            X=style_features(texts)
            self.clf=HistGradientBoostingClassifier(max_iter=220,learning_rate=.06,max_leaf_nodes=15,
                min_samples_leaf=30,l2_regularization=10,max_bins=127,early_stopping=False,random_state=SEED).fit(X,y)
            self.scale=max(float(np.std(self.clf.decision_function(X))),1e-6)
            return self
        normalized=[normalize(t) for t in texts]
        if self.kind=='normalized_nb':fit_text=normalized
        elif self.kind=='augmented_nb':fit_text=texts+normalized;y=np.r_[y,y]
        elif self.kind=='raw_nb':fit_text=texts
        else:raise ValueError(self.kind)
        self.features=FeatureUnion([
            ('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True)),
            ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=150000,sublinear_tf=True))])
        X=self.features.fit_transform(fit_text).tocsr();binary=(X>0).astype(float)
        counts=[1+np.asarray(binary[y==c].sum(axis=0)).ravel() for c in (0,1)]
        self.ratio=np.log((counts[1]/counts[1].sum())/(counts[0]/counts[0].sum()))
        X=X.multiply(self.ratio).tocsr()
        weights=np.full(len(y),.5) if self.kind=='augmented_nb' else None
        self.clf=LinearSVC(C=.5,random_state=SEED).fit(X,y,sample_weight=weights)
        self.scale=max(float(np.std(self.clf.decision_function(X))),1e-6)
        return self

    def _score(self,texts):
        return self.clf.decision_function(self.features.transform(texts).multiply(self.ratio).tocsr())

    def decision_function(self,texts):
        texts=list(texts)
        if self.kind=='style':return self.clf.decision_function(style_features(texts))/self.scale
        if self.kind=='normalized_nb':score=self._score([normalize(t) for t in texts])
        elif self.kind=='augmented_nb':score=(self._score(texts)+self._score([normalize(t) for t in texts]))/2
        else:score=self._score(texts)
        return score/self.scale


class BlendModel:
    """Fixed-weight blend of separately training-standardized component margins."""
    def __init__(self,kind='raw_style25'):
        options={'raw_style25':('raw_nb',.25),'aug_style25':('augmented_nb',.25),'aug_style50':('augmented_nb',.5)}
        if kind not in options:raise ValueError(kind)
        self.kind=kind;self.lexical_kind,self.style_weight=options[kind]

    def fit(self,texts,y):
        self.lexical=RobustModel(self.lexical_kind).fit(texts,y)
        self.stylistic=RobustModel('style').fit(texts,y)
        return self

    def decision_function(self,texts):
        return (1-self.style_weight)*self.lexical.decision_function(texts)+self.style_weight*self.stylistic.decision_function(texts)
