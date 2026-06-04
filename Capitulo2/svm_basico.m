%% SVM primal - separable, básico
clear all; close all; clc;

% 1. Generación de datos aleatorios linealmente separables
rng(42); % Semilla para reproducibilidad

% Parámetros
numero_puntos = 50;
centro_pos = [1.2, 1.2];  
centro_neg = [-1.2, -1.2];
varianza = 0.7;

datos_pos = varianza*randn(numero_puntos, 2) + centro_pos;
datos_neg = varianza*randn(numero_puntos, 2) + centro_neg;
datos = [datos_pos; datos_neg];
etiquetas = [ones(numero_puntos, 1); -ones(numero_puntos, 1)];

% 2. Llamar la función SVM básica primal
[w, b] = svm_prim_sep(datos, etiquetas);

% 3. Visualización de los resultados
figure('Color', 'w', 'Units', 'inches','Position', [2, 2, 7, 5], 'Name', 'SVM Básico - Hard Margin');
hold on; grid on;
set(gca, 'FontSize', 16, 'TickLabelInterpreter', 'latex');

% Dibujar puntos
scatter(datos_neg(:,1), datos_neg(:,2), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
scatter(datos_pos(:,1), datos_pos(:,2), 40, 'r', 'x', 'LineWidth', 1.2);

% Definir límites para el gráfico
x_lims = [min(datos(:,1))-0.5, max(datos(:,1))+0.5];
y_lims = [min(datos(:,2))-0.5, max(datos(:,2))+0.5];
x_range = linspace(x_lims(1), x_lims(2), 100);

% Ecuación del hiperplano: w1*x + w2*y + b = 0  => y = -(w1*x + b) / w2
f_hiperplano = @(x) -(w(1)*x + b) / w(2);
f_margen_pos = @(x) -(w(1)*x + b - 1) / w(2);
f_margen_neg = @(x) -(w(1)*x + b + 1) / w(2);

% Dibujar hiperplano y márgenes
plot(x_range, f_hiperplano(x_range), 'k-', 'LineWidth', 1);
plot(x_range, f_margen_neg(x_range), 'b--', 'LineWidth', 1);
plot(x_range, f_margen_pos(x_range), 'r--', 'LineWidth', 1);


% 4. Vectores de soporte: label * (w'x + b) es approx 1
distancias = etiquetas .* (datos * w + b);
indices_sv = find(abs(distancias - 1) < 1e-3);
scatter(datos(indices_sv, 1), datos(indices_sv, 2), 100, 'k', 'LineWidth', 1.2);

% 5. Vector normal w
x_w = sum(x_lims)/2;
y_w = f_hiperplano(x_w);
w_norm = w / norm(w); % Normalizar
quiver(x_w, y_w, w_norm(1), w_norm(2), 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
text(x_w + w(1)*0.3, y_w + w(2)*0.1, '$w$', 'Interpreter', 'latex', 'FontSize', 18,  'Color', [0.5 0.5 0.5]);

%
title('SVM Básico', 'FontSize', 18);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 18);
axis([x_lims y_lims]);
legend('Clase -1', 'Clase +1', 'Hiperplano', 'Margen -1', 'Margen +1', 'Vectores Soporte', 'Interpreter', 'latex', 'Location', 'northeastoutside', 'FontSize', 16);
axis equal tight;
hold off;

%% Función SVM básico - forma primal
function [w, b] = svm_prim_sep(data, labels)
% entrada:
%   data: matriz nxm; n: número de puntos, m: dimensión de un punto
%   labels: vector nx1 - clase a que cada punto pertenece: +1 / -1
% salida:
%   w: vector nx1. dirección normal del hiperplano
%   b: escalar. el sesgo

    [num, dim] = size(data);
    cvx_begin 
        variables w(dim) b;
        minimize(norm(w));
        subject to
            labels .* (data * w + b) >= 1;
    cvx_end
end