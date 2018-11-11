%Declaration des cours
cours(inf1005c). %classe A
cours(inf1500). %classe B
cours(inf1010). %classe D
cours(log1000). %classe C
cours(inf1600). %classe C
cours(inf2010). %classe F
cours(log2410). %classe E
cours(inf2705). %classe D
cours(inf1900). %classe C
cours(log2810). %classe F
cours(mth1007). %classe G
cours(log2990). %classe G
cours(inf2205). %classe C

%Declaration des prerequis direct de cours, tels que representes sur le graphe
prerequis(log1000, inf1005c).
prerequis(inf1010, inf1005c).
prerequis(inf1600, inf1005c).
prerequis(inf1600, inf1500).
prerequis(log2410, log1000).
prerequis(log2410, inf1010).
prerequis(inf2010, inf1010).
prerequis(inf2705, inf2010).

%Declaration des classes d'équivalence de cours
classe(a).
classe(b).
classe(c).
classe(d).
classe(e).
classe(f).
classe(g).

%Assignation des cours à leur classe d'équivalance
classeAppartenance(a, inf1005c).
classeAppartenance(b, inf1500).
classeAppartenance(c, log1000).
classeAppartenance(c, inf1600).
classeAppartenance(c, inf2205).
classeAppartenance(c, inf1900).
classeAppartenance(d, inf1010).
classeAppartenance(e, log2410).
classeAppartenance(f, log2810).
classeAppartenance(f, inf2010).
classeAppartenance(g, inf2705).
classeAppartenance(g, mth1007).
classeAppartenance(g, log2990).

%Declaration des liens de prerequis entre les classes
classePrerequis(g,f).
classePrerequis(f,d).
classePrerequis(e,d).
classePrerequis(e,c).
classePrerequis(d,a).
classePrerequis(c,a).
classePrerequis(c,b).

%Pour avoir les cours corequis, on regarde si deux cours appartiennent a la meme classe
corequis(C1, C2) :- classeAppartenance(X,C1) , classeAppartenance(X,C2) , C1 \= C2.

%Un cours est prerequis a un autre si ils sont corequis, ou si ils ont un lien prerequis direct
prerequisComplet(C1, C2) :- corequis(C1,C2) ; prerequis(C1,C2).

%Retourne vrai si le cours est dans la classe ou bien dans une classe parente à la classe initiale
classeRequis(Classe, C):- classeAppartenance(Classe, C) ; (classePrerequis(Classe, ClassePrerequise), classeRequis(ClassePrerequise, C)).

%Retourne vrai si les cours sont dans le même classe et différents ou si C2 est un dans une classe parente a la classe de C1
toutRequis(C1, C2) :- corequis(C1,C2) ; (classeAppartenance(ClasseC1, C1), classePrerequis(ClasseC1, ClasseC2), classeRequis(ClasseC2, C2)).

courAPrendreComplet(C, L) :- setof(X, toutRequis(C, X), L).