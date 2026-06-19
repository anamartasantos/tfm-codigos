%% Predicción SVM All-together de Crammer & Singer
% Autoría: Directores del proyecto con adaptación
% Contexto: Herramienta utilizada en el Capítulo 3 del TFM de Ana Marta Oliveira dos Santos
% Nota: Este código es una implementación base proporcionada por los tutores 
% del proyecto. La única modificación realizada respecto al 
% script original ha sido la sustitución de la función de evaluación final 
% para integrar la métrica personalizada del proyecto (medidas_completas.m).

function [Loss,Bal_accu,Sol] = MultiSVM_CS_cvx(Xtest,Yot,X,Y,FunPara)
% Model:
%   min_{W,b,xi}  (1/2)||W||_F^2 + C * sum_{i=1}^m xi_i
%
% s.a.
%   w_{y_i}^T x_i + b_{y_i} - (w_k^T x_i + b_k) >= 1 - xi_i,   ∀ k ≠ y_i
%   xi_i >= 0
%
% Este modelo corresponde a la formulación multiclass "all-together"
% propuesta por Crammer & Singer (2001), donde se utiliza una única
% variable de holgura por muestra (xi_i), en contraste con el modelo
% de Weston–Watkins que usa una holgura por clase rival (xi_{ik}).

[m,n] = size(X);

% ===== Mapear etiquetas a 1,...,K =====
classes = unique(Y);
K = numel(classes);

Yidx = zeros(m,1);
for i = 1:m
    Yidx(i) = find(classes == Y(i), 1);
end

C = FunPara.c;

% ===== Entrenamiento =====
tic;
cvx_begin quiet
    cvx_precision('low')
    cvx_solver sedumi  
    variables W(n,K) b(1,K) Xi(m)
    minimize( 0.5 * sum(sum(W.^2)) + C * sum(Xi) )
    subject to
        Xi >= 0;
        for i = 1:m
            yi = Y(i);
            for k = 1:K
                if k ~= yi
                    X(i,:)*W(:,yi) + b(yi) - X(i,:)*W(:,k) - b(k) >= 1 - Xi(i);
                end
            end
        end
cvx_end

Tf = toc;

% ===== Guardar solución =====
Sol.W = W;
Sol.b = b;
Sol.Xi = Xi;
Sol.Tf = Tf;
Sol.cvx_status = cvx_status;
Sol.cvx_optval = cvx_optval;

% ===== Scores y predicción =====
scores = Xtest*W + repmat(b,size(Xtest,1),1);
[~, ypred_idx] = max(scores, [], 2);

% Etiquetas originales predichas
ypred = classes(ypred_idx);
Sol.Ypred = ypred;
Sol.scores = scores;

% ===== Convertir a codificación one-hot =====
mt = size(Xtest,1);
Prediction = zeros(mt,K);
for j = 1:mt
    Prediction(j,ypred_idx(j)) = 1;
end

% ===== Medidas =====
Loss = NaN; 
[~, Bal_accu, ~, ~, ~] = medidas_completas(Prediction, Yot, K);

end
