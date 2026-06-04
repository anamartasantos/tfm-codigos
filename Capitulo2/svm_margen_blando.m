%% SVM Soft Margin - Margen Blando - forma primal
clear all; close all; clc;

% 1. Generación de datos con solapamiento controlado
rng(10); % Semilla para reproducibilidad

% Parámetros
numero_puntos = 35;
varianza = 0.8;

datos_pos = varianza*randn(numero_puntos, 2) + 1.2; 
datos_neg = varianza*randn(numero_puntos, 2)- 0.2;
datos = [datos_pos; datos_neg];
etiquetas = [ ones(numero_puntos, 1); -ones(numero_puntos, 1)];

% 2. Parámetro C - C alto hace el margen más estrecho
C = 1; 

% 3. Entrenamiento
[w, b] = svm_prim_nonsep(datos, etiquetas, C);

% 4. Visualización de los resultados
figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 8, 5], 'Name', 'SVM Margen Blando');
hold on; grid on;
set(gca, 'FontSize', 16, 'TickLabelInterpreter', 'latex');

% Dibujar puntos
scatter(datos_neg(:,1), datos_neg(:,2), 40, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
scatter(datos_pos(:,1), datos_pos(:,2), 40, 'r', 'Marker', 'x', 'LineWidth', 1.2);

% Límites para el gráfico
x_lim = [min(datos(:,1))-1, max(datos(:,1))+1];
y_lim = [min(datos(:,2))-1, max(datos(:,2))+1];
[X, Y] = meshgrid(linspace(x_lim(1), x_lim(2), 200), linspace(y_lim(1), y_lim(2), 200));
Z = w(1)*X + w(2)*Y + b;

% Regiones
contourf(X, Y, sign(Z), 'LineStyle', 'none', 'FaceAlpha', 0.15, 'HandleVisibility','off');
colormap([0.7 0.8 1; 1 0.7 0.7]); 

% Frontera y márgenes
contour(X, Y, Z, [0 0], 'k-', 'LineWidth', 2);      % Frontera
contour(X, Y, Z, [-1 -1], 'b--', 'LineWidth', 1.2); % Margen -
contour(X, Y, Z, [1 1], 'r--', 'LineWidth', 1.2);   % Margen +

% Vectores de Soporte
distancias = etiquetas .* (datos * w + b);
indices_sv = find(distancias <= 1.01); 
scatter(datos(indices_sv, 1), datos(indices_sv, 2), 100, 'k', 'LineWidth', 1);

% 
title(['SVM Margen Blando con C = ', num2str(C)], 'Interpreter', 'latex', 'FontSize', 16);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 18);
axis([x_lim y_lim]);
legend('Clase -1', 'Clase +1', 'Hiperplano', 'Margen -1', 'Margen +1', 'Vectores Soporte','Interpreter', 'latex', 'Location', 'northeastoutside', 'FontSize', 16);
axis equal tight;
hold off;

%% Función Primal SVM soft margin
function [w, b] = svm_prim_nonsep(data, labels, C)
% entrada:
%   data: matriz nxm; n: número de puntos, m: dimensión de un punto
%   labels: vector nx1 - clase a que cada punto pertenece: +1 / -1
% salida:
%   w: vector nx1. dirección normal del hiperplano
%   b: escalar. el sesgo

    [num, dim] = size(data);
    cvx_begin quiet
        variables w(dim) b xi(num);
        minimize(sum(w.^2)/2 + C*sum(xi) );
        subject to
            labels .* (data * w + b) >= 1 - xi;
            xi >= 0;
    cvx_end
end