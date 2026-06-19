%% Función de evaluación de métricas - multiclase
% Autor: Ana Marta Oliveira dos Santos
% Fecha: Junio de 2026
% Contexto: Capítulo 3 del Trabajo de Fin de Máster
% Nota: Implementación propia adaptada a partir del código base proporcionado 
% por los directores del proyecto.
%
% Descripción: Calcula Accuracy, Balanced Accuracy, Precision, Recall y F1-Score (Macro-promediadas).
% Input:
%       Prediction - Etiqueta predicción (matriz de tamaño mt x T)
%       Yot        - Etiqueta real (matriz de tamaño mt x T) 
%       T          - Número de clases  
% Output:
%       Acc        - Accuracy global
%       Bacc       - Balanced Accuracy
%       Precision  - Precisión (Macro)
%       Recall     - Exhaustividad (Macro)
%       F1         - F1-Score (Macro)

function [Acc, Bacc, Precision, Recall, F1] = medidas_completas(Prediction, Yot, T)
    % Convierte matrices one-hot (o +1/-1) a vectores de índices
    [~, y_pred] = max(Prediction, [], 2);
    [~, y_true] = max(Yot, [], 2);
    
    % 1. Accuracy global
    Acc = sum(y_true == y_pred) / length(y_true);
    
    % Inicializar métricas por clase
    prec_class = zeros(T, 1);
    rec_class = zeros(T, 1);
    f1_class = zeros(T, 1);
    bacc_class = zeros(T, 1);
    
    % Calcular métricas para cada clase (k)
    for k = 1:T
        TP = sum((y_pred == k) & (y_true == k));
        FP = sum((y_pred == k) & (y_true ~= k));
        FN = sum((y_pred ~= k) & (y_true == k));
        
        % Recall y Bacc component
        if sum(y_true == k) > 0
            rec_class(k) = TP / sum(y_true == k);
            bacc_class(k) = rec_class(k);
        else
            rec_class(k) = 1;
            bacc_class(k) = 1;
        end
        
        % Precisión
        if (TP + FP) > 0
            prec_class(k) = TP / (TP + FP);
        else
            prec_class(k) = 0; % Si no predijo esta clase, precisión 0
        end
        
        % F1-Score
        if (prec_class(k) + rec_class(k)) > 0
            f1_class(k) = 2 * (prec_class(k) * rec_class(k)) / (prec_class(k) + rec_class(k));
        else
            f1_class(k) = 0;
        end
    end
    
    % 2. Balanced Accuracy
    Bacc = mean(bacc_class);
    
    % 3. Precisión, Recall y F1 macro-promediada
    Precision = mean(prec_class);
    Recall = mean(rec_class);
    F1 = mean(f1_class);
end