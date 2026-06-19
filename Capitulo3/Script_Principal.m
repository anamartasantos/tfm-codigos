%% Script principal TFM: Evaluación robusta y multiescenario (paralelizado)
% Autor: Ana Marta Oliveira dos Santos
% Fecha: Junio de 2026
% Contexto: Capítulo 3 del Trabajo de Fin de Máster
% Descripción: Ejecuta escenarios limpios y con inyección de perturbaciones
% estocásticas (etiquetas y características). Utiliza parfor para reducir 
% el tiempo de cómputo. Realiza validación cruzada y test estadístico (Wilcoxon).
%
% Nota sobre autoría: 
% Este script principal ha sido desarrollado íntegramente por la autora. 
% Sin embargo, las funciones de entrenamiento y predicción invocadas en el 
% bloque "switch modelo_actual" (Predi_OVA_SVM, Predi_OVO_SVM, MultiSVM_WW_cvx, 
% MultiSVM_CS_cvx, SVM_soft_quadsolve) son implementaciones base proporcionadas 
% por los directores del proyecto.

clear; clc; close all;
warning('off'); 

% Rutas y dependencias
addpath(genpath('./SVM'));
addpath(genpath('./Multi_SVM'));
addpath(genpath('./Dataset_Multiclase')); 

% Configuración experimental
datasets = {'irisMn', 'hayes_roth', 'wineMn', 'glassMn'};
nombres_ds = {'Iris', 'Hayes-Roth', 'Wine', 'Glass'};
modelos = {'OvA', 'OvO', 'All_WW', 'All_CS'}; 
escenarios = {'Base', 'Label_5', 'Label_10', 'Feature_5', 'Feature_10'};
num_runs = 10; 
folds = 5; 
C_values = 2.^(-3:2:5); 
nC = length(C_values);

% Pre-asignación de matrices 4D para parfor
BACC_matrix = NaN(length(datasets), length(escenarios), length(modelos), num_runs);
F1_matrix   = NaN(length(datasets), length(escenarios), length(modelos), num_runs);
Time_matrix = NaN(length(datasets), length(escenarios), length(modelos), num_runs);

Resultados = cell(length(escenarios), 1);
for s = 1:length(escenarios)
    Resultados{s} = table();
end

fprintf('=====================================================\n');
fprintf(' Iniciando experimentación TFM (%d runs en paralelo) \n', num_runs);
fprintf('=====================================================\n\n');

%% Bucle principal
for d = 1:length(datasets)
    fprintf('>> Analizando Dataset: [%s]...\n', nombres_ds{d});
    load(datasets{d}); 
    
    [m, n] = size(X);
    T = max(Y); 
    
    % Copias locales seguras para mandar a los procesadores paralelos
    X_data = X; Y_data = Y; Yo_data = Yo;
    
    %% Inicio del bucle paralelo
    parfor run = 1:num_runs
        warning('off');
        fprintf('   Ejecutando Run %d/%d...\n', run, num_runs);
        
        % Matriz temporal exclusiva para este run, de modo a evitar conflitos entre procesadores
        bacc_run = NaN(length(escenarios), length(modelos));
        f1_run   = NaN(length(escenarios), length(modelos));
        time_run = NaN(length(escenarios), length(modelos));
        
        % Parámetros de la SVM - copia aislada para este procesador
        FunPara_local = struct();
        FunPara_local.kerfPara.type = 'lin';
        
        % 1. Partición de datos idéntica y bloqueada por semilla
        rng(42 + run, 'twister'); 
        idx_perm = randperm(m);
        num_test = round(0.20 * m);
        idx_test = idx_perm(1:num_test);
        idx_train = idx_perm(num_test+1:end);
        
        X_tr_base = X_data(idx_train, :); Y_tr_raw = Y_data(idx_train, :); Yo_tr_raw = Yo_data(idx_train, :);
        X_ts_base = X_data(idx_test, :); Yo_ts_raw = Yo_data(idx_test, :);
        
        m_tr = size(X_tr_base, 1);
        m_ts = size(X_ts_base, 1);
        
        %% Bucle de escenarios
        for s = 1:length(escenarios)
            escenario_actual = escenarios{s};
            
            X_tr = X_tr_base; Y_tr = Y_tr_raw; Yo_tr = Yo_tr_raw;
            X_ts = X_ts_base; Yo_ts = Yo_ts_raw;
            
            if strcmp(nombres_ds{d}, 'Hayes-Roth') && contains(escenario_actual, 'Feature')
                continue; 
            end
            
            % Inyección de ruido
            if strcmp(escenario_actual, 'Label_5') || strcmp(escenario_actual, 'Label_10')
                ruido = str2double(strrep(escenario_actual, 'Label_', '')) / 100;
                num_ruido = round(ruido * m_tr);
                idx_ruido = randperm(m_tr, num_ruido);
                
                for idx_n = idx_ruido
                    clase_original = Y_tr(idx_n);
                    clases_disp = setdiff(1:T, clase_original);
                    nueva_clase = clases_disp(randi(length(clases_disp)));
                    
                    Y_tr(idx_n) = nueva_clase;
                    val_pos = Yo_tr(idx_n, clase_original);
                    val_neg = Yo_tr(idx_n, nueva_clase);
                    Yo_tr(idx_n, clase_original) = val_neg;
                    Yo_tr(idx_n, nueva_clase) = val_pos;
                end
                
            elseif strcmp(escenario_actual, 'Feature_5') || strcmp(escenario_actual, 'Feature_10')
                ruido = str2double(strrep(escenario_actual, 'Feature_', '')) / 100;
                std_tr = std(X_tr_base, 0, 1);
                
                X_tr = X_tr + (ruido * repmat(std_tr, m_tr, 1) .* randn(size(X_tr)));
                X_ts = X_ts + (ruido * repmat(std_tr, m_ts, 1) .* randn(size(X_ts)));
            end
            
            %% Bucle de modelos
            for mod_idx = 1:length(modelos)
                modelo_actual = modelos{mod_idx};
                
                % Tuning interno - 5-fold cv
                BACCU_cv = zeros(nC,1);
                for c_idx = 1:nC
                    FunPara_local.c = C_values(c_idx);
                    bacc_fold = zeros(folds,1);
                    idx_cv = randperm(m_tr);
                    
                    for k = 1:folds
                        tst_cv = idx_cv(k:folds:m_tr);
                        trn_cv = setdiff(1:m_tr, tst_cv);
                        
                        Xa_raw_fold = X_tr(trn_cv,:);
                        Xt_cv_raw_fold = X_tr(tst_cv,:);
                        Ya = Y_tr(trn_cv,:);
                        Yot_cv = Yo_tr(tst_cv,:);
                        
                        % Normalización rigurosa: calculada solo con X_train para evitar data leakage hacia X_test
                        min_fold = min(Xa_raw_fold);
                        max_fold = max(Xa_raw_fold);
                        den_fold = max_fold - min_fold;
                        den_fold(den_fold < 1e-8) = 1; 
                        
                        Xa = 2 * ((Xa_raw_fold - repmat(min_fold, size(Xa_raw_fold,1), 1)) ./ repmat(den_fold, size(Xa_raw_fold,1), 1)) - 1;
                        Xt_cv = 2 * ((Xt_cv_raw_fold - repmat(min_fold, size(Xt_cv_raw_fold,1), 1)) ./ repmat(den_fold, size(Xt_cv_raw_fold,1), 1)) - 1;
                        
                        try
                            switch modelo_actual
                                case 'OvA', [~, bacc_fold(k), ~, ~] = Predi_OVA_SVM(Xt_cv, Yot_cv, Xa, Ya, FunPara_local, T);
                                case 'OvO', [~, bacc_fold(k), ~] = Predi_OVO_SVM(Xt_cv, Yot_cv, Xa, Ya, FunPara_local, T);
                                case 'All_WW', [~, bacc_fold(k), ~] = MultiSVM_WW_cvx(Xt_cv, Yot_cv, Xa, Ya, FunPara_local);
                                case 'All_CS', [~, bacc_fold(k), ~] = MultiSVM_CS_cvx(Xt_cv, Yot_cv, Xa, Ya, FunPara_local);
                            end
                        catch
                            bacc_fold(k) = NaN;
                        end
                    end
                    BACCU_cv(c_idx) = mean(bacc_fold, 'omitnan');
                end
                
                [~, bestIdx] = max(BACCU_cv);
                mejor_C = C_values(bestIdx);
                
                % Entrenamiento final y evaluación
                FunPara_local.c = mejor_C;
                t0_test = tic;
                
                % Normalización final para Test
                min_final = min(X_tr);
                max_final = max(X_tr);
                den_final = max_final - min_final;
                den_final(den_final < 1e-8) = 1;
                
                X_tr_norm = 2 * ((X_tr - repmat(min_final, size(X_tr,1), 1)) ./ repmat(den_final, size(X_tr,1), 1)) - 1;
                X_ts_norm = 2 * ((X_ts - repmat(min_final, size(X_ts,1), 1)) ./ repmat(den_final, size(X_ts,1), 1)) - 1;
                
                try
                    switch modelo_actual
                        case 'OvA'
                            [~, ~, ~, Sol] = Predi_OVA_SVM(X_ts_norm, Yo_ts, X_tr_norm, Y_tr, FunPara_local, T);
                            Fk_Test = X_ts_norm * Sol.W + repmat(Sol.Bias, size(X_ts_norm,1), 1);
                            [~, ypred_idx] = max(Fk_Test, [], 2);
                        case 'OvO'
                            mt_ts = size(X_ts_norm,1); cont_k = zeros(T, mt_ts);
                            for i = 1:T
                                fin1 = find(Y_tr==i); A = X_tr_norm(fin1,:); mi = length(fin1);
                                for j = i+1:T
                                    fin2 = find(Y_tr==j); B = X_tr_norm(fin2,:); mj = length(fin2);
                                    [Predict_Y, ~] = SVM_soft_quadsolve([A;B], [ones(mi,1); -ones(mj,1)], X_ts_norm, FunPara_local);
                                    for kl = 1:mt_ts
                                        if Predict_Y(kl) == 1, cont_k(i,kl) = cont_k(i,kl) + 1;
                                        else, cont_k(j,kl) = cont_k(j,kl) + 1; end
                                    end
                                end
                            end
                            [~, ypred_idx] = max(cont_k, [], 1); ypred_idx = ypred_idx';
                        case 'All_WW'
                            [~, ~, Sol] = MultiSVM_WW_cvx(X_ts_norm, Yo_ts, X_tr_norm, Y_tr, FunPara_local);
                            [~, ypred_idx] = max(Sol.scores, [], 2);
                        case 'All_CS'
                            [~, ~, Sol] = MultiSVM_CS_cvx(X_ts_norm, Yo_ts, X_tr_norm, Y_tr, FunPara_local);
                            [~, ypred_idx] = max(Sol.scores, [], 2);
                    end
                    
                    Prediction = zeros(size(X_ts_norm,1), T);
                    for p_idx = 1:length(ypred_idx), Prediction(p_idx, ypred_idx(p_idx)) = 1; end
                    
                    [~, Bacc, ~, ~, F1] = medidas_completas(Prediction, Yo_ts, T);
                catch
                    Bacc = NaN; F1 = NaN;
                end
                
                % Guardamos los resultados de este run en las variables temporales aisladas
                bacc_run(s, mod_idx) = Bacc;
                f1_run(s, mod_idx) = F1;
                time_run(s, mod_idx) = toc(t0_test);
                
            end % fin modelos
        end % fin escenarios
        
        % Almacenamos los resultados en las matrices globales maestras
        BACC_matrix(d, :, :, run) = bacc_run;
        F1_matrix(d, :, :, run)   = f1_run;
        Time_matrix(d, :, :, run) = time_run;
        
    end % Fin del bucle parfor
    
    %% Consolidación de resultados por datset
    for s = 1:length(escenarios)
        if strcmp(nombres_ds{d}, 'Hayes-Roth') && contains(escenarios{s}, 'Feature')
            continue; 
        end
        for mod_idx = 1:length(modelos)
            bacc_vec = squeeze(BACC_matrix(d, s, mod_idx, :));
            f1_vec   = squeeze(F1_matrix(d, s, mod_idx, :));
            time_vec = squeeze(Time_matrix(d, s, mod_idx, :));
            
            str_bacc = sprintf('%.4f +- %.4f', mean(bacc_vec, 'omitnan'), std(bacc_vec, 'omitnan'));
            str_f1 = sprintf('%.4f +- %.4f', mean(f1_vec, 'omitnan'), std(f1_vec, 'omitnan'));
            mean_time = mean(time_vec, 'omitnan');
            
            nueva_fila = {nombres_ds{d}, modelos{mod_idx}, str_bacc, str_f1, mean_time};
            Resultados{s}(end+1, :) = nueva_fila;
        end
    end
end

%% Análisis estadístico
fprintf('\nCalculando Significancia Estadística...\n');
TestEstadistico = table();
idx_CS = find(strcmp(modelos, 'All_CS'));
idx_WW = find(strcmp(modelos, 'All_WW'));
idx_OvO = find(strcmp(modelos, 'OvO'));

for d = 1:length(datasets)
    for s = 1:length(escenarios)
        if strcmp(nombres_ds{d}, 'Hayes-Roth') && contains(escenarios{s}, 'Feature')
            continue; 
        end
        runs_CS = squeeze(BACC_matrix(d, s, idx_CS, :));
        runs_WW = squeeze(BACC_matrix(d, s, idx_WW, :));
        runs_OvO = squeeze(BACC_matrix(d, s, idx_OvO, :));
        
        try p_CS_vs_WW = signrank(runs_CS, runs_WW); catch, p_CS_vs_WW = NaN; end
        try p_CS_vs_OvO = signrank(runs_CS, runs_OvO); catch, p_CS_vs_OvO = NaN; end
        
        sig_WW = 'No'; if p_CS_vs_WW < 0.05, sig_WW = 'Sí'; end
        sig_OvO = 'No'; if p_CS_vs_OvO < 0.05, sig_OvO = 'Sí'; end
        
        fila_test = {nombres_ds{d}, escenarios{s}, p_CS_vs_WW, sig_WW, p_CS_vs_OvO, sig_OvO};
        TestEstadistico(end+1, :) = fila_test;
    end
end
TestEstadistico.Properties.VariableNames = {'Dataset', 'Escenario', 'P_Valor_CS_vs_WW', 'CS_Mejora_WW', 'P_Valor_CS_vs_OvO', 'CS_Mejora_OvO'};

%% Exportación
archivo_excel = 'Resultados_TFM_Completo.xlsx';
if exist(archivo_excel, 'file'), delete(archivo_excel); end
for s = 1:length(escenarios)
    if ~isempty(Resultados{s})
        Resultados{s}.Properties.VariableNames = {'Dataset', 'Modelo', 'Balanced_Accuracy', 'F1_Score', 'Mean_Time_seg'};
        writetable(Resultados{s}, archivo_excel, 'Sheet', escenarios{s});
    end
end
writetable(TestEstadistico, archivo_excel, 'Sheet', 'Test_Estadistico');
fprintf('\n Experimentos completados. Resultados guardados en: %s\n', archivo_excel);