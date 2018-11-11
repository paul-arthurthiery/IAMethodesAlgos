/* ---------------------------------------------------------- */
/* ---------------Definition Question------------------------ */
/* ---------------------------------------------------------- */



/*First Level*/
ask(energie ,X):-
  format('X a besoin d energie ?',[X]),
  read(Reponse),
  Reponse ='oui'.

/*Second Level*/
ask(plante ,X):-
  format('X est une plante ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(menage ,X):-
  format('X sert au menage ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(communiquer ,X):-
  format('X sert a communiquer ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(cuisine ,X):-
  format('X est dans la cusine ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(eclairer ,X):-
  format('X sert a eclairer ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(laver ,X):-
  format('X sert a laver ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(main ,X):-
  format('X tient dans la main ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(musique ,X):-
  format('X fait de la musique ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(remplir ,X):-
  format('X peut se faire remplir ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(poche ,X):-
  format('X peut tenir dans une poche ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(dormir ,X):-
  format('X peut on dormir dessus ?',[X]),
  read(Reponse),
  Reponse ='oui'.

/*Third Level*/

ask(meuble ,X):-
  format('X est un meuble ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(cafe ,X):-
  format('X peut servir a faire du cafe ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(corps ,X):-
  format('X peut servir se laver une partie du corps ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(manger ,X):-
  format('X sert a manger?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(piece ,X):-
  format('X peut contenir des pieces ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(ranger ,X):-
  format('On peut ranger des choses dans X ?',[X]),
  read(Reponse),
  Reponse ='oui'.

/* ---------------------------------------------------------- */
/* ----------------Definition Niveau------------------------ */
/* ---------------------------------------------------------- */

/*First Level*/

objet(X):-
  ask(energie, X),
  energieSecondLevel(X).
objet(X):-
  noEnergieSecondLevel(X).

/*Second Level*/

energieSecondLevel(X):-
  ask(menage ,X),
  menage(X),
  energie(X).

energieSecondLevel(X):-
  ask(communiquer, X),
  communiquer(X),
  energie(X).

energieSecondLevel(X):-
  ask(cuisine, X),
  cuisine(X),
  energie(X).

energieSecondLevel(X):-
  ask(eclairer, X),
  eclairer(X),
  energie(X).

noEnergieSecondLevel(X):-
  ask(ranger, X),
  ranger(X),
  noEnergie(X).

noEnergieSecondLevel(X):-
  ask(plante, X),
  plante(X),
  noEnergie(X).

noEnergieSecondLevel(X):-
  ask(laver, X),
  laver(X),
  noEnergie(X).

noEnergieSecondLevel(X):-
  ask(cuisine, X),
  cusine(X),
  noEnergie(X).

noEnergieSecondLevel(X):-
  ask(main, X),
  main(X),
  noEnergie(X).

noEnergieSecondLevel(X):-
  ask(musique, X),
  musique(X),
  noEnergie(X).

noEnergieSecondLevel(X):-
  ask(dormir, X),
  dormir(X),
  noEnergie(X).

noEnergieSecondLevel(X):-
  ask(poche, X),
  poche(X),
  noEnergie(X).

noEnergieSecondLevel(X):-
  noPoche(X),
  noEnergie(X).

/*Third Level*/

communiquer(X):-
  ask(main, X),
  main(X).
communiquer(X):-
  noMain(X).

cuisine(X):-
  ask(meuble, X),
  meuble(X),
  energie(X).
cuisine(X):-
  noMeuble(X),
  energie(X).

cuisine(X):-
  ask(manger, X),
  manger(X),
  noEnergie(X).
cuisine(X):-
  noManger(X),
  noEnergie(X).

laver(X):-
  ask(corps, X),
  corps(X).
laver(X):-
  ask(corps, X),
  noCorps(X).

poche(X):-
  ask(piece, X),
  piece(X).
poche(X):-
  noArgent(X).


/*Fourth Level*/

meuble(X):-
  ask(remplir, X),
  remplir(X).
meuble(X):-
  noRemplir(X).

noMeuble(X):-
  ask(cafe, X),
  cafe(X).
noMeuble(X):-
  noCafe(X).

manger(X):-
  ask(main, X),
  main(X).
manger(X):-
  noMain(X).

noManger(X):-
  ask(remplir, X),
  remplir(X).
noManger(X):-
  noRemplir(X).




menage('Aspirateur').
noMain('Ordinateur').
noMain('Balai').
noMain('Assiette').
main('Téléphone').
main('Fourchette').
plante('Cactus').
remplir('Four').
remplir('Casserole').
ranger('Sac à dos').
noRemplir('Cuisinière').
noRemplir('Table').
cafe('Cafetière').
noCafe('Grille-pain').
corps('Shampooing').
noCorps('Détergent à vaisselle').
domrir('Lit').
musique('Piano').
eclairer('Lampe').
noPoche('Papier').
piece('Portefeuille').
noArgent('Clé').

energie('Aspirateur').
energie('Ordinateur').
energie('Téléphone').
energie('Four').
energie('Cuisinière').
energie('Cafetière').
energie('Grille-pain').
energie('Lampe').

noEnergie('Fourchette').
noEnergie('Balai').
noEnergie('Cactus').
noEnergie('Assiette').
noEnergie('Table').
noEnergie('Casserole').
noEnergie('Shampooing').
noEnergie('Détergent à vaisselle').
noEnergie('Lit').
noEnergie('Clé').
noEnergie('Sac à dos').
noEnergie('Piano').
noEnergie('Papier').
noEnergie('Portefeuille').
