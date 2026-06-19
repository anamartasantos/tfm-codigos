%% Función de Soporte - Kernel
% Autoría: Directores del proyecto / Código de terceros proporcionado
% Contexto: Herramienta matemática de soporte utilizada en el TFM de Ana Marta Oliveira dos Santos.
% Nota: Este código es una implementación matemática base facilitada por los 
% tutores del proyecto. Se incluye sin modificaciones en este repositorio 
% con el único fin de garantizar la reproducibilidad de los experimentos.
% -------------------------------------------------------------------------

function omega = kernelfun(Xtrain,kerfPara,Xt)
% Construct the positive (semi-) definite and symmetric kernel matrix
%
% >> Omega = kernelfun(X, kernel_fct)
%
% This matrix should be positive definite if the kernel function
% satisfies the Mercer condition. Construct the kernel values for
% all test data points in the rows of Xt, relative to the points of X.
%
% >> Omega_Xt = kernelfun(X, kernel_fct, Xt)
%
%
% Full syntax
%
% >> Omega = kernelfun(X, kernel_fct)
% >> Omega = kernelfun(X, kernel_fct, Xt)
%
% Outputs
%   Omega  : N x N (N x Nt) kernel matrix
% Inputs
%   X      : N x d matrix with the inputs of the training data
%   kerfPara.type : Kernel type (by default 'RBF_kernel')
%   kerfPara.pars   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%   Xt(*)  : Nt x d matrix with the inputs of the test data

kernel_type = kerfPara.type;
kernel_pars = kerfPara.pars;

nb_data = size(Xtrain,1);


if strcmp(kernel_type,'rbf'),
    if nargin<3,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./(2*kernel_pars(1)));
    else
        omega = - 2*Xtrain*Xt';
        Xtrain = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        Xt = sum(Xt.^2,2)*ones(1,nb_data);
        omega = omega + Xtrain+Xt';
        omega = exp(-omega./(2*kernel_pars(1)));
    end
  
elseif strcmp(kernel_type,'lin')
    if nargin<3,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly')
    if nargin<3,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
end
