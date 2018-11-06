cours(inf1005c).
cours(inf1500).
cours(inf1010).
cours(log1000).
cours(inf1600).
cours(inf2010).
cours(log2410).
cours(inf2705).
cours(inf1900).
cours(log2810).
cours(mth1007).
cours(log2990).
cours(inf2205).

prerequis(log1000, inf1005c).
prerequis(inf1010, inf1005c).
prerequis(inf1600, inf1005c).
prerequis(inf1600, inf1500).
prerequis(log2410, log1000).
prerequis(log2410, inf1010).
prerequis(inf2010, inf1010).
prerequis(inf2705, inf2010).

corequis(inf2010, log2810).
corequis(inf2705, mth1007).
corequis(inf2705, log2990).
corequis(inf1900, inf1600).
corequis(inf1900, log1000).
corequis(inf1900, inf2205).

toutRequis(A, B) :- corequis(A, B) ; prerequis(A, B); ((corequis(A, X) ; prerequis(A, X)), toutRequis(X, B)).

courAPrendreComplet(C, L) :- setof(X, toutRequis(C, X), L).