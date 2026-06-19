%% Predicción SVM Multi-clase OvO
% Autoría: Directores del proyecto con adaptación
% Contexto: Herramienta utilizada en el Capítulo 3 del TFM de Ana Marta Oliveira dos Santos
% Nota: Este código es una implementación base proporcionada por los tutores 
% del proyecto. La única modificación realizada respecto al 
% script original ha sido la sustitución de la función de evaluación final 
% para integrar la métrica personalizada del proyecto (medidas_completas.m).

function [Loss,Bal_accu,Tfinal]=Predi_OVO_SVM(Xt,Yot,Xa,Ya,FunPara,T)

mt=size(Xt,1);
Prediction=-ones(mt,T);
tf=zeros(T);
cont_k=zeros(T,mt);
for i=1:T
    fin1=find(Ya==i);
    mi=length(fin1);
    A=Xa(fin1,:);
    for j=i+1:T
        fin2=find(Ya==j);
        B=Xa(fin2,:);
        mj=length(fin2);
        Xtr=[A;B];
        Ytr=[ones(mi,1);-ones(mj,1)];
        [Predict_Y,tf(i,j)]=SVM_soft_quadsolve(Xtr,Ytr,Xt,FunPara);   
        clear w1 w2 b1 b2 
        for kl=1:mt
            if Predict_Y(kl)==1
               cont_k(i,kl)=cont_k(i,kl)+1;
            else
               cont_k(j,kl)=cont_k(j,kl)+1; 
            end
        end
        
    end
end
Tfinal=sum(sum(tf));
for j=1:mt 
    [max_vot,r]=max(cont_k(:,j));
    Prediction(j,r)=1;
end

Loss = NaN; 
[~, Bal_accu, ~, ~, ~] = medidas_completas(Prediction, Yot, T);

