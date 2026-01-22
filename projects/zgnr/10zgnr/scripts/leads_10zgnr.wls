#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*\[Beta][\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:=\[Beta][\[Omega],\[Delta],t,\[Epsilon],m]=ReplacePart[(\[Omega]+I*\[Delta]-\[Epsilon])IdentityMatrix[m],Join[Table[{i,i-1},{i,Range[2,m,1]}],Table[{i,i+1},{i,Range[1,m,1]}]]->-t]*)


(* ::Input:: *)
(*T[t_,m_]:=T[t,m]=ReplacePart[0*IdentityMatrix[m],Join[Table[{x,x+1},{x,Range[1,m,4]}],Table[{x,x-1},{x,Range[4,m,4]}]]->t]*)


(* ::Input:: *)
(*Clear[leadgenerateleft]*)


(* ::Input:: *)
(*leadgenerateleft[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:= leadgenerateleft[\[Omega],\[Delta],t,\[Epsilon],m]= Module[{unit=Inverse[\[Beta][\[Omega],\[Delta],t,\[Epsilon],m]],J=Inverse[\[Beta][\[Omega],\[Delta],t,\[Epsilon],m]]},Do[J= Inverse[IdentityMatrix[m]-unit.T[t,m].J.ConjugateTranspose[T[t,m]]].unit,50000];J=J]*)


(* ::Input:: *)
(*Show[ListLinePlot[Table[{\[Omega],-Im[leadgenerateleft[\[Omega],0.0001,1,0,10][[1,1]]]},{\[Omega],Range[0,3,0.01]}]]]*)


(* ::Input:: *)
(*SL[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:= Inverse[IdentityMatrix[m]-Inverse[\[Beta][\[Omega],\[Delta],t,\[Epsilon],m]].T[t,m].leadgenerateleft[\[Omega],\[Delta],t,\[Epsilon],m].ConjugateTranspose[T[t,m]]].Inverse[\[Beta][\[Omega],\[Delta],t,\[Epsilon],m]]*)


(* ::Input:: *)
(*SR[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:= Inverse[IdentityMatrix[m]-Inverse[\[Beta][\[Omega],\[Delta],t,\[Epsilon],m]].T[t,m].leadgenerateleft[\[Omega],\[Delta],t,\[Epsilon],m].ConjugateTranspose[T[t,m]]].Inverse[\[Beta][\[Omega],\[Delta],t,\[Epsilon],m]]*)


(* ::Input:: *)
(*(*SR[\[Omega]_]:= Inverse[IdentityMatrix[12]-Inverse[\[Beta][\[Omega],0.0001,1,0,12]].T[1,12].leadgenerateleft[\[Omega],0.0001,1,0,12].ConjugateTranspose[T[1,12]]].Inverse[\[Beta][\[Omega],0.0001,1,0,12]]*)*)


(* ::Input:: *)
(*Show[ListLinePlot[Table[{\[Omega],-Im[SR[\[Omega],0.0001,1,0,10][[1,1]]]},{\[Omega],Range[0,3,0.01]}]]]*)


(* ::Input:: *)
(*IL[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:= Inverse[IdentityMatrix[m]-SL[\[Omega],\[Delta],t,\[Epsilon],m].ConjugateTranspose[T[t,m]].SR[\[Omega],\[Delta],t,\[Epsilon],m].ConjugateTranspose[T[t,m]]].SL[\[Omega],\[Delta],t,\[Epsilon],m]*)


(* ::Input:: *)
(*IR[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:= Inverse[IdentityMatrix[m]-SR[\[Omega],\[Delta],t,\[Epsilon],m].ConjugateTranspose[T[t,m]].SL[\[Omega],\[Delta],t,\[Epsilon],m].ConjugateTranspose[T[t,m]]].SR[\[Omega],\[Delta],t,\[Epsilon],m]*)


(* ::Input:: *)
(*gdd[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:= IL[\[Omega],\[Delta],t,\[Epsilon],m]-ConjugateTranspose[IL[\[Omega],\[Delta],t,\[Epsilon],m]]*)


(* ::Input:: *)
(*grr[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:= IR[\[Omega],\[Delta],t,\[Epsilon],m]-ConjugateTranspose[IR[\[Omega],\[Delta],t,\[Epsilon],m]]*)


(* ::Input:: *)
(*Gnonlocal[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:=SR[\[Omega],\[Delta],t,\[Epsilon],m].ConjugateTranspose[T[t,m]].IL[\[Omega],\[Delta],t,\[Epsilon],m]*)


(* ::Input:: *)
(*GNON[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:= Gnonlocal[\[Omega],\[Delta],t,\[Epsilon],m]-ConjugateTranspose[Gnonlocal[\[Omega],\[Delta],t,\[Epsilon],m]]*)


(* ::Input:: *)
(*tr1[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,m_]:= Abs[Tr[gdd[\[Omega],\[Delta],t,\[Epsilon],m].T[t,m].grr[\[Omega],\[Delta],t,\[Epsilon],m].ConjugateTranspose[T[t,m]]-T[t,m].GNON[\[Omega],\[Delta],t,\[Epsilon],m].ConjugateTranspose[T[t,m]].GNON[\[Omega],\[Delta],t,\[Epsilon],m]]]*)


(* ::Input:: *)
(*ListLinePlot[Table[{\[Omega],tr1[\[Omega],0.0001,1,0,10]},{\[Omega],Range[0,3,0.01]}]]*)


(* ::Input:: *)
(*ListLinePlot[Table[{\[Omega],tr1[\[Omega],0.0001,1,0,50]},{\[Omega],Range[0,3,0.01]}]]*)


(* ::Input:: *)
(*F[n_]:= {n,n}*)


(* ::Input:: *)
(*unit[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,\[Epsilon]1_,n_]:=Module[{b=\[Beta][\[Omega],\[Delta],t,\[Epsilon],10]},ReplacePart[b,Table[F[RandomSample[{RandomInteger[{1,10}]},1]]//Flatten,n]->\[Omega]+I*\[Delta]-\[Epsilon]1]]*)


(* ::Input:: *)
(*device[\[Omega]_,\[Delta]_,t_,\[Epsilon]_,\[Epsilon]1_,m_,num1_,num2_,n1_,n2_]:=Module[{list,b2},*)
(*list={RandomSample[Join[Table[unit[\[Omega],0.0001,1,0,\[Epsilon]1,n1],num1],Table[unit[\[Omega],0.0001,1,0,\[Epsilon]1,n2],num2] ,Table[unit[\[Omega],0.0001,1,0,0,1],100-num2-num1]]]};*)
(*sl1= Module[{J=SL[\[Omega],0.0001,1,0,m]},Do[J=Inverse[IdentityMatrix[m]-Inverse[list[[1,\[Zeta]]]].T[t,m].J.ConjugateTranspose[T[t,m]]].Inverse[list[[1,\[Zeta]]]],{\[Zeta],1,100}];*)
(* J=J];*)
(*Il1:=Inverse[IdentityMatrix[m]-sl1.ConjugateTranspose[T[t,m]].leadgenerateleft[\[Omega],0.0001,1,0,m].ConjugateTranspose[T[t,m]]].sl1;*)
(*Ir1:=Inverse[IdentityMatrix[m]-leadgenerateleft[\[Omega],0.0001,1,0,m].ConjugateTranspose[T[t,m]].sl1.ConjugateTranspose[T[t,m]]].leadgenerateleft[\[Omega],0.0001,1,0,m];gdd1:=Il1-ConjugateTranspose[Il1];grr1:=Ir1-ConjugateTranspose[Ir1];Gnonlocal1:= SR[\[Omega],0.0001,1,0,m].ConjugateTranspose[T[t,m]].Il1;GNON1:= Gnonlocal1-ConjugateTranspose[Gnonlocal1];If[Abs[Tr[gdd1.T[t,m].grr1.ConjugateTranspose[T[t,m]]-T[t,m].GNON1.ConjugateTranspose[T[t,m]].GNON1]]>tr1[\[Omega],0.0001,1,0,m],tr1[\[Omega],0.0001,1,0,m],Abs[Tr[gdd1.T[t,m].grr1.ConjugateTranspose[T[t,m]]-T[t,m].GNON1.ConjugateTranspose[T[t,m]].GNON1]]]]*)


(* ::Input:: *)
(*ListLinePlot[Table[{\[Omega],device[\[Omega],0.0001,1,0,0.5,10,10,0,1,0]},{\[Omega],Range[0,3,0.01]}]]*)


(* ::Input:: *)
(*Mean[ParallelTable[device[0,0.0001,1,0,0.5,10,50,0,1,0],2000]]*)


(* ::Input:: *)
(*(*Table[Print["Working on === "<>ToString[num]<>"...."];Export["/home/shardulmukim/PhD/fwi/zgnr/10zgnr/CA_"<>ToString[num]<>".DAT",Table[{\[Omega],Mean[ParallelTable[device[\[Omega],0.0001,1,0,0.5,10,num,0,1,0],2000]]},{\[Omega],Range[0,3,0.01]}]],{num,Range[5,95,5]}]*)*)


(* ::Input:: *)
(*transmission=Table[Import["/home/shardulmukim/PhD/fwi/zgnr/10zgnr/CA_"<>ToString[x]<>".DAT"],{x,Range[5,95,5]}]*)
