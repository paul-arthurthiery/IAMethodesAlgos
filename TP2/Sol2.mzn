include "N14a.dzn";
include "alldifferent.mzn";
int: nbTeams;
int: N = nbTeams;
array[1..N, 1..N] of var int: pv;%location of each match


set of int: teamNumbers = 1..N;  % our calendar has N teams
set of int: days = 1..N-1; % N-1 days are need to play all matches

array[days, teamNumbers] of var teamNumbers: calendar;%each line is a day, each column is a team. each element is a tuple [team playing against column, isAHomeMatch]


predicate oneMatchPerTeamPerDay(int: day) = alldifferent([calendar[day, team] | team in teamNumbers]);%a team cannot play more than once a day
predicate noRepetingMatches(int: team) = alldifferent([calendar[day, team] | day in days]);%every match must be different

%global constraints for our matrix
constraint forall(i in 1..N-1, j in 1..N)(calendar[i, j]>=1 /\ calendar[i, j]<=N /\calendar[i, j] != j );%all teams have numbers between 1 and N, teams cannot play against themselves, matches are symetrical
constraint forall(i in 1..N-1)(oneMatchPerTeamPerDay(i));
constraint forall(i in 1..N)(noRepetingMatches(i));
constraint forall(j in 1..N) (
  forall(i in 1..N-5)(
    (pv[j,calendar[i,j]]+pv[j,calendar[i+1,j]]+pv[j,calendar[i+2,j]]+pv[j,calendar[i+3,j]] != 4) /\ (pv[j,calendar[i,j]]+pv[j,calendar[i+1,j]]+pv[j,calendar[i+2,j]]+pv[j,calendar[i+3,j]] != 0)
  )
);%making sure that the last 4 matches of the observed team are neither all at  or away 

solve satisfy;

output [ show(calendar[i,j])
           ++ (if(j mod N == 0) then " \n " else " " endif)
           | i in days, j in teamNumbers
        ];