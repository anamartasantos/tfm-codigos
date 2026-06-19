%% SVM Multiclase - Enfoque One-versus-One (OvO) y Resultado final
% Autor: Ana Marta Oliveira dos Santos
% Fecha: Junio de 2026
% Contexto: Capítulo 2 del Trabajo de Fin de Máster
% Descripción: Ilustración del enfoque One-versus-One (OvO) para clasificación
% multiclase mediante SVM, utilizando votación por pares y el criterio max-wins

clear all; close all; clc;

% 1. Generación de datos para 3 clases (A, B, C)
rng(10); 
n = 40; % Puntos por clase

% Generamos nubes de puntos con cierta dispersión
dataA = [2.0, 2.0] + randn(n, 2) * 0.6;   % Clase 1 (A)
dataB = [1.5, 5.0] + randn(n, 2) * 0.6;   % Clase 2 (B)
dataC = [5.0, 3.5] + randn(n, 2) * 0.7;   % Clase 3 (C)

X_total = [dataA; dataB; dataC];
y_total = [ones(n,1); 2*ones(n,1); 3*ones(n,1)];

% Parámetros de visualización y entrenamiento
C_param = 1;
colores = [0 0.447 0.741; 0.85 0.325 0.098; 0.929 0.694 0.125]; % Azul, Naranja, Amarillo
nombres = {'A', 'B', 'C'};
parejas = [1, 2; 1, 3; 2, 3]; % Las 3 combinaciones posibles K(K-1)/2

% Variables para guardar los modelos de cada par
W_models = zeros(size(parejas, 1), 2);
B_models = zeros(size(parejas, 1), 1);

% 2. Configuración de la figura 1 - clasificadores binarios
figure1 = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 12, 4]);
t = tiledlayout(1, 3, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% 3. Bucle para generar los 3 clasificadores OVO
for p = 1:size(parejas, 1)
    c1 = parejas(p, 1); % Primera clase de la dupla
    c2 = parejas(p, 2); % Segunda clase de la dupla
    c_out = setdiff(1:3, [c1, c2]); % Clase que queda fuera en este modelo
    
    nexttile;
    hold on; grid on;
    
    % Filtrar datos: solo las dos clases involucradas
    idx_train = (y_total == c1 | y_total == c2);
    X_train = X_total(idx_train, :);
    y_train = y_total(idx_train);
    
    % Convertir a etiquetas binarias +1 y -1
    y_bin = ones(size(y_train));
    y_bin(y_train == c2) = -1; 
    
    % Entrenamiento
    [w, b] = svm_prim_nonsep(X_train, y_bin, C_param);
    
    % Guardar w y b para la figura final
    W_models(p, :) = w';
    B_models(p) = b;
    
    % Visualización de puntos
    idx_out = (y_total == c_out);
    scatter(X_total(idx_out,1), X_total(idx_out,2), 30, [0.8 0.8 0.8], 'filled', 'MarkerFaceAlpha', 0.1, 'HandleVisibility','off');
    
    scatter(X_total(y_total==c1,1), X_total(y_total==c1,2), 40, colores(c1,:), 'filled', 'MarkerEdgeColor', 'k');
    scatter(X_total(y_total==c2,1), X_total(y_total==c2,2), 40, colores(c2,:), 'filled', 'MarkerEdgeColor', 'k');
    
    % Dibujar frontera
    ax = [min(X_total(:,1))-1 max(X_total(:,1))+1 min(X_total(:,2))-1 max(X_total(:,2))+1];
    [XX, YY] = meshgrid(linspace(ax(1), ax(2), 100), linspace(ax(3), ax(4), 100));
    ZZ = w(1)*XX + w(2)*YY + b;
    
    contourf(XX, YY, sign(ZZ), 'LineStyle', 'none', 'FaceAlpha', 0.1, 'HandleVisibility','off');
    contour(XX, YY, ZZ, [0 0], 'k-', 'LineWidth', 2); 
    
    title(['Clasificador ', nombres{c1}, ' vs ', nombres{c2}], 'Interpreter', 'latex', 'FontSize', 14);
    xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 16); ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 16);
    if p == 1
        legend(['Clase ', nombres{c1}], ['Clase ', nombres{c2}], 'Frontera $f_{kl}(x)$', 'Interpreter', 'latex', 'Location', 'northeast');
    end
    axis(ax); axis equal;
end

title(t, '\textbf{Enfoque One-versus-One (OvO): Votaci\''on por Pares}', 'Interpreter', 'latex', 'FontSize', 16);

% 4. Resultado clasificación final - max-wins
figure2 = figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 7, 5.5]);
hold on; grid on;

% Límites y malla
ax_final = [min(X_total(:,1))-1 max(X_total(:,1))+1 min(X_total(:,2))-1 max(X_total(:,2))+1];
[XX_final, YY_final] = meshgrid(linspace(ax_final(1), ax_final(2), 300), linspace(ax_final(3), ax_final(4), 300));
Puntos_Malla = [XX_final(:), YY_final(:)];
Num_Puntos = size(Puntos_Malla, 1);

% Matriz para ir sumando los votos de cada clase
Votos = zeros(Num_Puntos, 3);

% Evaluar cada modelo y acumular los votos según el signo de la frontera
for p = 1:size(parejas, 1)
    c1 = parejas(p, 1);
    c2 = parejas(p, 2);
    
    % Evaluar función de decisión f_kl(z)
    Z_malla = Puntos_Malla * W_models(p, :)' + B_models(p);
    
    % Votación: si Z > 0, suma 1 voto la clase 1 de la pareja. Si Z < 0, la clase 2.
    Votos(Z_malla > 0, c1) = Votos(Z_malla > 0, c1) + 1;
    Votos(Z_malla <= 0, c2) = Votos(Z_malla <= 0, c2) + 1; 
end

% Aplicar criterio max-wins: argmax sobre la suma de votos
[~, Clase_Predicha] = max(Votos, [], 2);

% Malla con las predicciones
ZZ_final = reshape(Clase_Predicha, size(XX_final));

% fondo
contourf(XX_final, YY_final, ZZ_final, 'LineStyle', 'none', 'FaceAlpha', 0.2);
colormap(figure2, colores);

% Puntos de datos originales
for k = 1:3
    idx = (y_total == k);
    scatter(X_total(idx,1), X_total(idx,2), 50, colores(k,:), 'filled', 'MarkerEdgeColor', 'k', ...
        'DisplayName', ['Clase ', nombres{k}]);
end

% Detalles del gráfico final
title('\textbf{Resultado Final de Clasificaci\''on (Regiones OvO)}', 'Interpreter', 'latex', 'FontSize', 16);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 16); 
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 16);
legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', 14);
axis(ax_final); axis equal;

%% Función Primal SVM soft margin
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