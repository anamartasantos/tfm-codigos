%% Predicción SVM Multi-clase OvA
% Autoría: Directores del proyecto con adaptación
% Contexto: Herramienta utilizada en el Capítulo 3 del TFM de Ana Marta Oliveira dos Santos
% Nota: Este código es una implementación base proporcionada por los tutores 
% del proyecto. La única modificación realizada respecto al 
% script original ha sido la sustitución de la función de evaluación final 
% para integrar la métrica personalizada del proyecto (medidas_completas.m).

function [Loss,Bal_accu,Tfinal,Sol]=Predi_OVA_SVM(Xt,Yot,Xa,Ya,FunPara,T)

m=size(Xa,1);
mt=size(Xt,1);
Prediction=-ones(mt,T);
tf=zeros(T,1);

for k=1:T
    fin1=find(Ya==k); 
    fin2=setdiff([1:length(Ya)]',[fin1]);
    mk=length(fin1);
    mkk=m-mk;
    A=Xa(fin1,:); % class k
    B=Xa(fin2,:); % Other classes
    Xtr=[A;B];
    Ytr=[ones(mk,1);-ones(mkk,1)];
    [~,tf(k),Sol]=SVM_soft_quadsolve(Xtr,Ytr,Xt,FunPara);
    Fk_Test(:,k)=Sol.Val_Xt;
    W(:,k)=Sol.w;
    Bias(k)=Sol.b;
end
Sol.W=W;
Sol.Bias=Bias;
Tfinal=sum(tf);

for j=1:mt
    [max_fk,rk]=max(Fk_Test(j,:));
    clear max_fk
    Prediction(j,rk)=1;
end

Loss = NaN; 
[~, Bal_accu, ~, ~, ~] = medidas_completas(Prediction, Yot, T);


