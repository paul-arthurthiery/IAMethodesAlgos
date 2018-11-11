/* ---------------------------------------------------------- */
/* ---------------Definition Question------------------------ */
/* ---------------------------------------------------------- */


/* Troisieme Niveau*/
ask(homme ,X):-
  format('X est un Homme ?',[X]),
  read(Reponse),
  Reponse ='oui'.

/* Quatrieme Niveau*/
ask(politique ,X):-
  format('X est relié à la politique ?',[X]),
  read(Reponse),
  Reponse ='oui'.
ask(cinema ,X):-
  format('X est relié au domaine cinematographique ?',[X]),
  read(Reponse),
  Reponse ='oui'.
ask(litterature ,X):-
  format('X est relié au domaine litteraire ?',[X]),
  read(Reponse),
  Reponse ='oui'.
ask(art ,X):-
  format('X est relié au domaine artistique ?',[X]),
  read(Reponse),
  Reponse ='oui'.
ask(sport ,X):-
  format('X est relié au domaine sportif ?',[X]),
  read(Reponse),
  Reponse ='oui'.
ask(religion, X):-
  format('X est relié au domaine religieux',[X]),
  read(Reponse),
  Reponse = 'oui'.
ask(jeuxVideo, X):-
  format('X est relié au domaine du jeux video',[X]),
  read(Reponse),
  Reponse = 'oui'.
ask(reine, X):-
  format('X est une reine ? ',[X]),
  read(Response),
  Response = 'oui'.
/* Cinquieme Niveau */
ask(guerreFroide, X) :-
  format('X est lié à la guerre froide ? ',[X]),
  read(Reponse),
  Reponse = 'oui'.


ask(createur, X) :-
  format('X est un createur de jeux video ? ',[X]),
  read(Reponse),
  Reponse = 'oui'.

ask(acteur ,X):-
  format('X est un acteur ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(realisateur ,X):-
  format('X est un acteur ?',[X]),
  read(Reponse),
  Reponse ='oui'.

ask(nouveauTestament, X):-
  format('X est un personnage du Nouveau Testament ?',[X]),
  read(Reponse),
  Reponse = 'oui'.

ask(personnage, X):-
  format('X est un personnage ficitif ?',[X]),
  read(Reponse),
  Reponse = 'oui'.

ask(chanteur, X):-
  format('X est un chanteur ?',[X]),
  read(Reponse),
  Reponse = 'oui'.

ask(streetart, X):-
  format('X fait du streetart ?',[X]),
  read(Reponse),
  Reponse = 'oui'.

ask(secondeGuerre, X):-
  format('X est lié à la seconde guerre mondiale ?',[X]),
  read(Reponse),
  Reponse = 'oui'.

ask(usa, X):-
  format('X est américain ?',[X]),
  read(Reponse),
  Reponse = 'oui'.

ask(vivant,X):-
  format('~w est vivant ?',[X]),
  read(Reponse),
  Reponse ='oui'.


/* ---------------------------------------------------------- */
/* __---------------Definition Niveau------------------------ */
/* ---------------------------------------------------------- */



/*First Level*/

personne(X):-
  ask(homme, X),
  hommeSecondLevel(X).

personne(X):-
  femmeSecondLevel(X).

/*Second Level*/

hommeSecondLevel(X):-
 ask(politique,X),
 politique(X),
 homme(X).

hommeSecondLevel(X):-
 ask(cinema,X),
 cinema(X),
 homme(X).

hommeSecondLevel(X):-
 ask(litterature,X),
 litterature(X),
 homme(X).

hommeSecondLevel(X):-
 ask(art,X),
 art(X),
 homme(X).

hommeSecondLevel(X):-
 ask(sport,X),
 sport(X),
 homme(X).

hommeSecondLevel(X):-
ask(religion,X),
religion(X),
homme(X).

hommeSecondLevel(X):-
ask(jeuxVideo,X),
jeuxVideo(X),
homme(X).
/**/

femmeSecondLevel(X):-
 ask(politique,X),
 politique(X),
 femme(X).

femmeSecondLevel(X):-
 ask(cinema,X),
 cinema(X),
 femme(X).

femmeSecondLevel(X):-
 ask(litterature,X),
 litterature(X),
 femme(X).

femmeSecondLevel(X):-
 ask(art,X),
 art(X),
 femme(X).

femmeSecondLevel(X):-
 ask(sport,X),
 sport(X),
 femme(X).

femmeSecondLevel(X):-
  ask(religion,X),
  religion(X),
  femme(X).

femmeSecondLevel(X):-
ask(jeuxVideo,X),
jeuxVideo(X),
femme(X).

femmeSecondLevel(X):-
ask(reine,X),
reine(X),
femme(X).

/* Third Level  */

politique(X):-
  ask(guerreFroide, X),
  guerreFroide(X).
politique(X):-
  ask(secondeGuerre, X),
  guerreFroide(X).

cinema(X):-
  ask(personnage, X),
  personnage(X).
cinema(X):-
  ask(acteur, X),
  acteur(X).
cinema(X):-
  ask(realisateur, X),
  realisateur(X).

art(X):-
  ask(chanteur, X),
  chanteur(X).
art(X):-
  ask(streetart, X),
  chanteur(X).

sport(X):-
    automobile(X).

religion(X):-
  ask(nouveauTestament,X),
  nouveauTestament(X).
religion(X):-
  ancienTestament(X).

jeuxVideo(X):-
  ask(personnage, X),
  personnage(X).
jeuxVideo(X):-
  ask(createur, X),
  createur(X).
/*Fourth Level*/

guerreFroide(X):-
  ask(usa, X),
  usa(X).
guerreFroide(X):-
  urss(X).

secondeGuerre(X):-
  ask(usa, X),
  usa(X).
secondeGuerre(X):-
  urss(X).

automobile(X):-
  ask(vivant, X),
  vivant(X).

automobile(X):-
  mort(X).
/* ---------------------------------------------------------- */
/* -----------------Definition Classification---------------- */
/* ---------------------------------------------------------- */




personnage('Lara Croft').
personnage('Mario').
personnage('James Bond').

createur('Hideo Kojima').


usa('Dwight D. Eisenhower').
usa('Richard Nixon').
urss('Mikhail Gorbachev').
urss('Joseph Staline').

realisateur('Quentin Tarantino').

acteur('Jennifer Lawrence').
acteur('Denzel Washington').

streetart('Bansky').

chanteur('Lady Gaga').

litterature('J. K. Rowling').
litterature('Victor Hugo').

reine('Cléopâtre').

nouveauTestament('Jesus').
nouveauTestament('Pape Francois').
ancienTestament('Moise').

mort('Ayrton Senna').
vivant('Fernando Alonso').

femme('Lara Croft').
femme('Jennifer Lawrence').
femme('Lady Gaga').
femme('J. K. Rowling').
femme('Cléopâtre').

homme('Mario').
homme('James Bond').
homme('Mikhail Gorbachev').
homme('Joseph Staline').
homme('Dwight D. Eisenhower').
homme('Richard Nixon').
homme('Quentin Tarantino').
homme('Denzel Washington').
homme('Banksy').
homme('Victor Hugo').
homme('Jesus').
homme('Moise').
homme('Ayrton Senna').
homme('Fernando Alonso').
homme('Pape Francois').
homme('Hideo Kojima').
