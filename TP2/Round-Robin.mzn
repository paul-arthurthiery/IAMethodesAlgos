include "N14a.dzn";
int: nbTeams;
int: N = nbTeams;
array[0..N-1,0..N-1] of var int: pv;


array[0..N-1, 0..6, 0..1] of var int: calendar;%each line is a day, each tuple is a match between the 2 teams
array[0..N-1, 0..1] of var int: matchCounter;%each tuple is a team, first one is the number of following matches, second one is the place of the previous match (home is 1, away is 0)

constraint forall(i in 0..N-1, j in 0..N-1 where pv[calendar[i, j, 0], calendar[i, j, 1]] == 0 /\ matchCounter[i, 1] == 0)(matchCounter[i, 0] = matchCounter[i, 0] + 1);
constraint forall(i in 0..N-1, j in 0..N-1 where pv[calendar[i, j, 0], calendar[i, j, 1]] == 1 /\ matchCounter[i, 1] == 1)(matchCounter[i, 0] = matchCounter[i, 0] + 1);
constraint forall(i in 0..N-1, j in 0..N-1 where pv[calendar[j, i, 0], calendar[j, i, 1]] == 1 /\ matchCounter[j, 1] == 0)(matchCounter[j, 0] = matchCounter[j, 0] + 1);
constraint forall(i in 0..N-1, j in 0..N-1 where pv[calendar[j, i, 0], calendar[j, i, 1]] == 0 /\ matchCounter[j, 1] == 1)(matchCounter[j, 0] = matchCounter[j, 0] + 1);
constraint forall(i in 0..N-1, j in 0..N-1 where pv[calendar[i, j, 0], calendar[i, j, 1]] == 0 /\ matchCounter[i, 1] == 1)(matchCounter[i, 0] = 0);
constraint forall(i in 0..N-1, j in 0..N-1 where pv[calendar[i, j, 0], calendar[i, j, 1]] == 1 /\ matchCounter[i, 1] == 0)(matchCounter[i, 0] = 0);
constraint forall(i in 0..N-1, j in 0..N-1 where pv[calendar[j, i, 0], calendar[j, i, 1]] == 1 /\ matchCounter[j, 1] == 1)(matchCounter[i, 0] = 0);
constraint forall(i in 0..N-1, j in 0..N-1 where pv[calendar[j, i, 0], calendar[j, i, 1]] == 0 /\ matchCounter[j, 1] == 0)(matchCounter[i, 0] = 0);
constraint forall(i in 0..N-1, j in 0..N-1)(matchCounter[i, j] <= 4);

solve satisfy;

output show(calendar[0,0,0]);