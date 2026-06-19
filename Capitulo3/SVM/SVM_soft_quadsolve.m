%% Función de Soporte - SVM Quadsolve]
% Autoría: Directores del proyecto
% Contexto: Herramienta matemática de soporte utilizada en el TFM de Ana Marta Oliveira dos Santos.
% Nota: Este código es una implementación matemática base facilitada por los 
% tutores del proyecto. Se incluye sin modificaciones en este repositorio 
% con el único fin de garantizar la reproducibilidad de los experimentos.
% -------------------------------------------------------------------------

% Solving the following Quadratic Problems with quadsolve solver

% Dual problem of SVM softmargin
%  minimize  0.5*x'*K*x -e'*x
%  subject to  Y*x=0
%          0<= x <= C


function [Ytest,Tf,Sol]=SVM_soft_quadsolve(X,Y,Xt,FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%

%       Input:
%               X       - Training Data matrix (Each row vector is a data point)
%               Y       - Training label vector
%               Xt      - Test Data matrix.

%               FunPara - Struct value in Matlab. The fields in options
%                         that can be set:
%                   c: [0,inf] Parameter to tune the weight.

%       Output:
%               Ytest  - Predict value of the Xt.
%.              Sol.Val_Xt - Value of the Xt

%       Example:
%           A = rand(50,10);
%           B = rand(60,10);
%           X=[A;B];
%           Y=[ones(50,1);-ones(60,1)];
%           TestX=rand(20,10);
%           FunPara.kerfPara.type = 'lin';
%           FunPara.c=2^(2);
%
%           Ytest=SVM_soft_quadsolve(X,Y,TestX,FunPara);

    C=FunPara.c;
    kerfPara = FunPara.kerfPara;
    % Compute Kernel 
    if strcmp(kerfPara.type,'lin')
       K = X*X';
    else
       K=kernelfun(X,kerfPara);
    end
    K=K.*(Y*Y');

    t0=cputime;
    [alpha,bias]= quadsolve(K,-ones(size(K,1),1),Y',0,C); 
    Tf=cputime-t0;
    clear K

    Sol.alpha=alpha;
    alpha=alpha.*Y;
    Sol.b=-bias;
    if strcmp(kerfPara.type,'lin')
        w=X'*alpha;
        Sol.Val_Xt=Xt*w-bias;
        Sol.w=w;
    else
        Kt=kernelfun(X,kerfPara,Xt);
        Sol.Val_Xt=Kt'*alpha-bias;
    end
    Ytest=sign(Sol.Val_Xt);
end
