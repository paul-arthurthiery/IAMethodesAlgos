include "alldifferent.mzn";

int: n = 5;%5 maisons 

var 1..n : anglais;
var 1..n : espagnol;
var 1..n : ukrainien;
var 1..n : norvegien;
var 1..n : japonais;
var 1..n : rouge;
var 1..n : verte;
var 1..n : blanche;
var 1..n : jaune;
var 1..n : bleue;
var 1..n : chien;
var 1..n : escargot;
var 1..n : renard;
var 1..n : cheval;
var 1..n : zebre;
var 1..n : cafe;
var 1..n : the;
var 1..n : lait;
var 1..n : jusdorange;
var 1..n : eau;
var 1..n : sculpteur;
var 1..n : diplomate;
var 1..n : medecin;
var 1..n : violoniste;
var 1..n : acrobate;

%array[anglais, espagnol, ukrainien, norvegien, japonais] of var int: nationalites;
%array[rouge, verte, blanche, jaune, bleue] of var int: couleurs;
%array[chien, escargot, renard, cheval, zebre] of var int: animaux;
%array[cafe, the, lait, jusdorange, eau] of var int: boissons;
%array[sculpteur, diplomate, medecin, violoniste, acrobate] of var int: metiers;

constraint anglais = rouge;
constraint espagnol = chien;
constraint verte = cafe;
constraint ukrainien = the;
constraint verte = blanche+1;
constraint sculpteur = escargot;
constraint diplomate = jaune;
constraint lait = 3;
constraint norvegien = 1;
constraint ((medecin = renard-1) \/ (medecin = renard+1));
constraint ((diplomate = cheval-1) \/ (diplomate = cheval+1));
constraint violoniste = jusdorange;
constraint japonais = acrobate;
constraint ((norvegien = bleue-1) \/ (norvegien = bleue+1));
constraint alldifferent([anglais, espagnol, ukrainien, norvegien, japonais]);
constraint alldifferent([rouge, verte, blanche, jaune, bleue]);
constraint alldifferent([chien, escargot, renard, cheval, zebre]);
constraint alldifferent([cafe, the, lait, jusdorange, eau]);
constraint alldifferent([sculpteur, diplomate, medecin, violoniste, acrobate]);

solve satisfy;

output ["celui qui boit de l'eau = ", show(eau), "\n", "celui qui possède le zèbre = ", show(zebre),"\n"]