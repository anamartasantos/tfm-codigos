%% Ilustración
%% SVM Multiclase - Enfoque One-versus-Rest (OvR) y Resultado Final
% Autor: Ana Marta Oliveira dos Santos
% Fecha: Junio de 2026
% Contexto: Capítulo 2 del Trabajo de Fin de Máster
% Descripción: Ilustración del enfoque One-versus-Rest (OvR) para clasificación
% multiclase mediante SVM, evaluando regiones de decisión mediante el criterio argmax.

clear all; close all; clc;

% 1. Generación de datos para 3 clases (A, B, C)
rng(10); 
n = 40; % Puntos por clase

dataA = [2.0, 2.0] + randn(n, 2) * 0.6;   % Clase A
dataB = [1.5, 5.0] + randn(n, 2) * 0.6;   % Clase B
dataC = [5.0, 3.5] + randn(n, 2) * 0.7;   % Clase C

X = [dataA; dataB; dataC];
% Etiquetas originales: 1, 2 y 3
y_orig = [ones(n,1); 2*ones(n,1); 3*ones(n,1)];

% Parámetros estéticos y de SVM
C_param = 1;
colores_clases = [0 0.447 0.741; 0.85 0.325 0.098; 0.929 0.694 0.125]; % Azul, Naranja, Amarillo
nombres = {'A', 'B', 'C'};

% Variables para guardar los modelos de cada clase
W_models = zeros(3, 2);
B_models = zeros(3, 1);

% 2. Configuración de la primera figura - 3 clasificadores
figure1 = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 12, 4]);
t = tiledlayout(1, 3, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% 3. Bucle para generar los 3 clasificadores OVR
for k = 1:3
    nexttile;
    hold on; grid on;
    
    % Crear etiquetas binarias: +1 para la clase k, -1 para el resto
    y_bin = -ones(size(y_orig));
    y_bin(y_orig == k) = 1;
    
    % Entrenamiento usando la función de margen blando (primal)
    [w, b] = svm_prim_nonsep(X, y_bin, C_param);
    
    % Guardar w y b
    W_models(k, :) = w';
    B_models(k) = b;
    
    % Puntos
    idx_rest = (y_orig ~= k);
    scatter(X(idx_rest,1), X(idx_rest,2), 30, [0.8 0.8 0.8], 'filled', 'MarkerFaceAlpha', 0.5);
    
    idx_one = (y_orig == k);
    scatter(X(idx_one,1), X(idx_one,2), 40, colores_clases(k,:), 'filled', 'MarkerEdgeColor', 'k');
    
    % Frontera de decisión
    ax = [min(X(:,1))-1 max(X(:,1))+1 min(X(:,2))-1 max(X(:,2))+1];
    [XX, YY] = meshgrid(linspace(ax(1), ax(2), 100), linspace(ax(3), ax(4), 100));
    ZZ = w(1)*XX + w(2)*YY + b;
    
    contourf(XX, YY, sign(ZZ), 'LineStyle', 'none', 'FaceAlpha', 0.05, 'HandleVisibility','off');
    colormap(gca, [0.9 0.9 0.9; colores_clases(k,:)]); 
    
    contour(XX, YY, ZZ, [0 0], 'k-', 'LineWidth', 2);      
    contour(XX, YY, ZZ, [1 1], 'LineStyle', '--', 'Color', colores_clases(k,:), 'LineWidth', 1); 
    contour(XX, YY, ZZ, [-1 -1], 'k--', 'LineWidth', 1);
    
    title(['Clase ', nombres{k}, ' vs Resto'], 'Interpreter', 'latex', 'FontSize', 12);
    if k == 1
        legend(['Resto (', nombres{2}, '+', nombres{3}, ')'], ['Clase ', nombres{k}], 'Frontera SVM', 'Interpreter', 'latex', 'Location', 'northeast');
    end
    xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 16); ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 16);
    axis(ax); axis equal;
end
title(t, '\textbf{Enfoque One-versus-Rest (OvR) para SVM Multiclase}', 'Interpreter', 'latex', 'FontSize', 16);

% 4. Resultado de clasificación final - argmax
figure2 = figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 7, 5.5]);
hold on; grid on;

% Límites y malla para evaluar las regiones
ax_final = [min(X(:,1))-1 max(X(:,1))+1 min(X(:,2))-1 max(X(:,2))+1];
[XX_final, YY_final] = meshgrid(linspace(ax_final(1), ax_final(2), 300), linspace(ax_final(3), ax_final(4), 300));
Puntos_Malla = [XX_final(:), YY_final(:)];

% Evaluar la función de decisión para los 3 modelos: (W * X) + B
% Dimensiones: Puntos_Malla(N x 2) * W_models'(2 x 3) + B_models'(1 x 3) = Puntuaciones(N x 3)
Puntuaciones = Puntos_Malla * W_models' + B_models';

% Criterio argmax: obtener el índice de la clase ganadora
[~, Clase_Predicha] = max(Puntuaciones, [], 2);

% Malla con predicciones
ZZ_final = reshape(Clase_Predicha, size(XX_final));

% color fondo
contourf(XX_final, YY_final, ZZ_final, 'LineStyle', 'none', 'FaceAlpha', 0.2);
colormap(figure2, colores_clases); % Asegurar que usa azul, naranja, amarillo

% Puntos originales
for k = 1:3
    idx = (y_orig == k);
    scatter(X(idx,1), X(idx,2), 50, colores_clases(k,:), 'filled', 'MarkerEdgeColor', 'k', ...
        'DisplayName', ['Clase ', nombres{k}]);
end

% Detalles gráfico final
title('\textbf{Resultado Final de Clasificaci\''on (Regiones de Decisi\''on OvR)}', 'Interpreter', 'latex', 'FontSize', 16);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 16); 
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 16);
legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', 14);
axis(ax_final); axis equal;

%% Función primal SVM soft margin
function [w, b] = svm_prim_nonsep(data, labels, C)
    [num, dim] = size(data);
    cvx_begin quiet
        variables w(dim) b xi(num);
        minimize(0.5 * sum(w.^2) + C * sum(xi));
        subject to
            labels .* (data * w + b) >= 1 - xi;
            xi >= 0;
    cvx_end
end