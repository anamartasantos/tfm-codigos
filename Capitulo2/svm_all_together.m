%% Script SVM Multi-clase: Comparativa Weston-Watkins vs Crammer-Singer
% Autor: Ana Marta Oliveira dos Santos
% Fecha: Junio de 2026
% Contexto: Capítulo 2 del Trabajo de Fin de Máster
% Descripción: Implementación y comparación de enfoques simultáneos (All-together) 
% para SVM multiclase, contrastando las variables de holgura de Weston & Watkins y Crammer & Singer.

clear; clc; close all;

% 1. Datos sintéticos
rng(10); 
n = 100;
data1 = [2.2, 1.8] + randn(n, 2) * 0.7;   % Clase 1: Azul
data2 = [1.2, 5.2] + randn(n, 2) * 0.8;   % Clase 2: Rojo
data3 = [4.2, 6.2] + randn(n, 2) * 1.0;   % Clase 3: Amarillo

X = [data1; data2; data3];
Y = [ones(n,1); 2*ones(n,1); 3*ones(n,1)];

C_param = 1;

% 2. Entrenamiento con las dos formulaciones, sin penalizar el sesgo
fprintf('Entrenando modelo Weston & Watkins\n');
[W_WW, b_WW] = svm_WW_cvx(X, Y, C_param);

fprintf('Entrenando modelo Crammer & Singer\n');
[W_CS, b_CS] = svm_CS_cvx(X, Y, C_param);

% 3. Figura
figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 12, 5.5]);
t = tiledlayout(1, 2, 'TileSpacing', 'Compact', 'Padding', 'Compact');
colores_fondo = [0.4 0.6 1; 1 0.4 0.4; 1 0.9 0.4]; 

% Malla para evaluación de fronteras
ax_lims = [-3 8 -1 10]; 
res = 800; 
x_range = linspace(ax_lims(1), ax_lims(2), res);
y_range = linspace(ax_lims(3), ax_lims(4), res);
[XX, YY] = meshgrid(x_range, y_range);
grid_pts = [XX(:), YY(:)];

% Formulación 1: Weston & Watkins
nexttile; hold on;
scores_WW = grid_pts * W_WW + b_WW; 
[~, map_WW] = max(scores_WW, [], 2);
map_WW = reshape(map_WW, size(XX));

imagesc(x_range, y_range, map_WW); alpha(0.15); colormap(gca, colores_fondo); set(gca, 'YDir', 'normal');
[~, ~] = contour(XX, YY, map_WW, [1.5 2.5], 'k', 'LineWidth', 2);

scatter(data1(:,1), data1(:,2), 40, 'o', 'MarkerEdgeColor', [0 0.3 0.8], 'LineWidth', 1.2);
scatter(data2(:,1), data2(:,2), 60, 'x', 'MarkerEdgeColor', [0.8 0 0], 'LineWidth', 1.5);
scatter(data3(:,1), data3(:,2), 40, 's', 'MarkerEdgeColor', [0.7 0.5 0], 'LineWidth', 1.2);

axis(ax_lims); grid on;
xlabel('Dimensi\''on 1', 'Interpreter', 'latex', 'FontSize', 16); ylabel('Dimensi\''on 2', 'Interpreter', 'latex', 'FontSize', 16);
title('\textbf{A. Formulaci\''on Weston \& Watkins}', 'Interpreter', 'latex', 'FontSize', 14);

% Formulación 2: Crammer & Singer
nexttile; hold on;
scores_CS = grid_pts * W_CS + b_CS; 
[~, map_CS] = max(scores_CS, [], 2);
map_CS = reshape(map_CS, size(XX));

imagesc(x_range, y_range, map_CS); alpha(0.15); colormap(gca, colores_fondo); set(gca, 'YDir', 'normal');
[~, ~] = contour(XX, YY, map_CS, [1.5 2.5], 'k', 'LineWidth', 2);

scatter(data1(:,1), data1(:,2), 40, 'o', 'MarkerEdgeColor', [0 0.3 0.8], 'LineWidth', 1.2);
scatter(data2(:,1), data2(:,2), 60, 'x', 'MarkerEdgeColor', [0.8 0 0], 'LineWidth', 1.5);
scatter(data3(:,1), data3(:,2), 40, 's', 'MarkerEdgeColor', [0.7 0.5 0], 'LineWidth', 1.2);

axis(ax_lims); grid on;
xlabel('Dimensi\''on 1', 'Interpreter', 'latex', 'FontSize', 16); ylabel('Dimensi\''on 2', 'Interpreter', 'latex', 'FontSize', 16);
title('\textbf{B. Formulaci\''on Crammer \& Singer}', 'Interpreter', 'latex', 'FontSize', 14);

title(t, '\textbf{Enfoques Simult\''aneos (All-together) para SVM Multiclase}', 'Interpreter', 'latex', 'FontSize', 16);

%% Funciones de optimización
function [W, b] = svm_WW_cvx(X, Y, C)
% Formulación Weston & Watkins: penaliza cada margen violado de forma independiente - matriz Xi
% Entradas: X (datos), Y (etiquetas), C (penalización)
% Salidas: W (pesos), b (sesgos)
    [m, n] = size(X);
    K = numel(unique(Y));
    cvx_begin quiet
        cvx_precision('low')
        variables W(n,K) b(1,K) Xi(m,K)
        minimize( 0.5 * sum(sum(W.^2)) + C * sum(sum(Xi)) )
        subject to
            Xi >= 0;
            for i = 1:m
                yi = Y(i);
                Xi(i,yi) == 0;
                for k = 1:K
                    if k ~= yi
                        X(i,:)*W(:,yi) + b(yi) - X(i,:)*W(:,k) - b(k) >= 2 - Xi(i,k);
                    end
                end
            end
    cvx_end
end

function [W, b] = svm_CS_cvx(X, Y, C)
% Formulación Crammer & Singer: penaliza solo la máxima violación por punto - vector Xi
% Entradas: X (datos), Y (etiquetas), C (penalización)
% Salidas: W (pesos), b (sesgos)
    [m, n] = size(X);
    K = numel(unique(Y));
    cvx_begin quiet
        cvx_precision('low')
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
end