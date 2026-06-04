%% SVMM Kernel - kernel gaussiano - dual
clear all; close all; clc;

% 1. Generación de datos
rng(15); % Semilla para reproducibilidad

% Parámetros
num_puntos_pos = 60;  % Puntos clase positiva
num_puntos_neg = 150; % Puntos clase negativa

% Clase +1 : datos centrales
datos_pos = 0.4 * randn(num_puntos_pos, 2); 

% Clase -1: datos exteriores
r_ext = 1.4 + 2.0 * rand(num_puntos_neg, 1); 
theta_ext = 2*pi * rand(num_puntos_neg, 1);
datos_neg = [r_ext .* cos(theta_ext), r_ext .* sin(theta_ext)];

% Combinar datos
datos = [datos_pos; datos_neg];
etiquetas = [ones(num_puntos_pos, 1); -ones(num_puntos_neg, 1)];

% 2. Parámetros del Kernel y SVM
C = 100;          
sigma = 1.2;

% 3. Entrenamiento - problema dual
[alpha_final, b, sv_idx] = svm_kernel_dual(datos, etiquetas, C, sigma);

% 4. Visualización 2D
figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 7, 6]);
hold on; grid on;
set(gca, 'FontSize', 14, 'TickLabelInterpreter', 'latex');

% Malla para la frontera
x_lim = [min(datos(:,1))-0.5, max(datos(:,1))+0.5];
y_lim = [min(datos(:,2))-0.5, max(datos(:,2))+0.5];
[X, Y] = meshgrid(linspace(x_lim(1), x_lim(2), 150), linspace(y_lim(1), y_lim(2), 150));

% Cálculo de la función de decisión
Z = zeros(size(X));
for i = 1:size(X,1)
    for j = 1:size(X,2)
        point = [X(i,j), Y(i,j)];
        k_vals = exp(-sum((datos - point).^2, 2) / (2 * sigma^2));
        Z(i,j) = sum(alpha_final .* etiquetas .* k_vals) + b;
    end
end

% Regiones de decisión
contourf(X, Y, sign(Z), 'LineStyle', 'none', 'FaceAlpha', 0.15, 'HandleVisibility','off');
colormap([0.75 0.85 1; 1 0.8 0.8]);

% Frontera y Márgenes
contour(X, Y, Z, [0 0], 'k-', 'LineWidth', 2.5);      % Hiperplano
contour(X, Y, Z, [1 1], 'r--', 'LineWidth', 1);       % Margen +1
contour(X, Y, Z, [-1 -1], 'b--', 'LineWidth', 1);     % Margen -1

% Puntos de datos
neg = scatter(datos_neg(:,1), datos_neg(:,2), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
pos = scatter(datos_pos(:,1), datos_pos(:,2), 40, 'r', 'x', 'LineWidth', 1.5);

% Vectores de Soporte
sv = scatter(datos(sv_idx, 1), datos(sv_idx, 2), 100, 'k', 'LineWidth', 1);

% 
title('\textbf{SVM con Kernel Gaussiano}', 'Interpreter', 'latex', 'FontSize', 16);
texto_subtitulo = sprintf('Par\\''ametros: $C = %g$, $\\sigma = %g$', C, sigma);
subtitle(texto_subtitulo, 'Interpreter', 'latex', 'FontSize', 14);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 16);
legend([neg, pos, sv], {'Clase $y=-1$', 'Clase $y=1$', 'Vectores Soporte'}, ...
    'Interpreter', 'latex', 'Location', 'southoutside', 'FontSize', 14);
axis equal tight;
hold off;


%%
% 5. Visualización 3D
figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 8, 6]);
hold on; grid on;
set(gca, 'FontSize', 14, 'TickLabelInterpreter', 'latex');

% 1. Calculamos la superficie de la "Mapping Function" (Suma de Kernels)
% Z_map = sum( alpha_i * y_i * K(xi, x) )
Z_map = zeros(size(X));
for i = 1:size(X,1)
    for j = 1:size(X,2)
        point = [X(i,j), Y(i,j)];
        k_vals = exp(-sum((datos - point).^2, 2) / (2 * sigma^2));
        Z_map(i,j) = sum(alpha_final .* etiquetas .* k_vals);
    end
end

% 2. Definimos el nivel del hiperplano de decisión (H' = 0 => Z = -b)
nivel_hiperplano = -b;

% 3. Crear Colormap binario Rojo/Azul basado exactamente en el umbral -b
% Para que la "cima" sea roja y la "base" sea azul
res = 256;
z_min = min(Z_map(:)); z_max = max(Z_map(:));
puntos_z = linspace(z_min, z_max, res);
mapa_dual = zeros(res, 3);
for k = 1:res
    if puntos_z(k) >= nivel_hiperplano
        mapa_dual(k,:) = [1, 0.3, 0.3]; % Rojo
    else
        mapa_dual(k,:) = [0.75 0.85 1]; % Azul
    end
end
colormap(mapa_dual);


% 4. Dibujar la superficie curva (la "montaña" de evidencia)
s = surf(X, Y, Z_map, 'EdgeColor', [0.4 0.4 0.4], 'EdgeAlpha', 0.1, 'FaceAlpha', 0.85, 'FaceColor', 'interp');

% 5. Hiperplano a altura -b
fill3([x_lim(1) x_lim(2) x_lim(2) x_lim(1)], [y_lim(1) y_lim(1) y_lim(2) y_lim(2)], ...
      [nivel_hiperplano nivel_hiperplano nivel_hiperplano nivel_hiperplano], ...
      [0.95 0.95 0.95], 'FaceAlpha', 0.1, 'EdgeColor', 'k', 'LineWidth', 1.5);
% Representación 3D del hiperplano/ frontera 2D
contour3(X, Y, Z_map, [nivel_hiperplano nivel_hiperplano], 'k', 'LineWidth', 2.5, 'HandleVisibility','off');

% 6. Proyectar puntos de datos sobre la superficie
Z_pos_map = zeros(num_puntos_pos, 1);
for i = 1:num_puntos_pos
    k_vals = exp(-sum((datos - datos_pos(i,:)).^2, 2) / (2 * sigma^2));
    Z_pos_map(i) = sum(alpha_final .* etiquetas .* k_vals);
end

Z_neg_map = zeros(num_puntos_neg, 1);
for i = 1:num_puntos_neg
    k_vals = exp(-sum((datos - datos_neg(i,:)).^2, 2) / (2 * sigma^2));
    Z_neg_map(i) = sum(alpha_final .* etiquetas .* k_vals);
end

% Puntos
scatter3(datos_neg(:,1), datos_neg(:,2), Z_neg_map, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
scatter3(datos_pos(:,1), datos_pos(:,2), Z_pos_map, 35, 'r', 'x', 'LineWidth', 1);


% 7. Estética final
view(-30, 20);
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 16);
zlabel('$\sum \alpha_i^* y_i K(x_i, x)$', 'Interpreter', 'latex', 'FontSize', 16);
title('\textbf{Superficie de la funci\''on de decisi\''on}', 'Interpreter', 'latex', 'FontSize', 16);
legend({'Superficie de Mapeo $\phi(x)$ (Relieve)', ...
        'Hiperplano de Separación ($Z = -b^*$)', ...
        'Clase $y = +1$ (Centrales)', ...
        'Clase $y = -1$ (Exteriores)'}, ...
        'Interpreter', 'latex', 'Location', 'southoutside', 'FontSize', 14);

camlight headlight;
lighting gouraud;
hold off;

%% Función Kernel SVM - Forma Dual
function [alpha_vec, b, sv_idx] = svm_kernel_dual(data, labels, C, sigma)
% entrada:
%   data: matriz nxm; n: número de puntos, m: dimensión de un punto
%   labels: vector nx1 - clase a que cada punto pertenece: +1 / -1
%   C :
%   sigma: parámetro del kernel gaussiano
% salida:
%   alpha_vec: vector nx1. variables duales
%   b: escalar. el sesgo


    num = size(data, 1);
    
    % Matriz de Kernel RBF
    dist_sq = sum(data.^2, 2) + sum(data.^2, 2)' - 2*(data * data');
    K = exp(-dist_sq / (2 * sigma^2));
    
    % Matriz H
    H = (labels * labels') .* K;
    
    cvx_begin quiet
        variable a_var(num)
        minimize( 0.5 * quad_form(a_var, H) - sum(a_var) )
        subject to
            a_var >= 0;
            a_var <= C;
            labels' * a_var == 0;
    cvx_end
    
    alpha_vec = a_var;
    sv_idx = find(alpha_vec > 1e-4);
    
    % Cálculo del b
    margin_idx = find(alpha_vec > 1e-4 & alpha_vec < C - 1e-4);
    if isempty(margin_idx), margin_idx = sv_idx(1); end
    b_values = labels(margin_idx) - K(margin_idx, :) * (alpha_vec .* labels);
    b = mean(b_values);
end