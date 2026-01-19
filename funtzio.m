function [dynamism1, dynamism2, dynamism3] = funtzio(filePath)

% Leer los datos desde un archivo Excel
data = readmatrix(filePath); % Alternativa: readtable(filePath)

% Asegurar que hay al menos dos columnas
if size(data, 2) < 2
    error('El archivo debe contener al menos dos columnas: tiempo y voltaje.');
end

% Extraer vectores por separado
time = data(:,1);      % Primera columna: tiempo
voltage = data(:,2);   % Segunda columna: voltaje

figure;
plot(time, voltage, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Voltage (V)');
title('Voltage profile');
grid on

rf = rainflow(voltage, time);

if isempty(rf)
    rf = zeros(1, 6);
    dynamism1 = [mean(voltage), 0, 10000000, 0, 0];
    dynamism2 = [median(voltage), 0, 10000000, 0, 0];
    dynamism3 = [mode(voltage), 0, 10000000, 0, 0];
    return
end

rf(:,6) = (rf(:,5) - rf(:,4)) ./ rf(:,1);
rf(:,7:11) = 0;

resultados = [];
inicio = 1;

for i = 2:length(voltage)
    if voltage(i) ~= voltage(i-1)
        if i - 1 > inicio
            resultados = [resultados; time(inicio), time(i-1), voltage(i-1)];
        end
        inicio = i;
    end
end

if length(voltage) >= inicio + 1
    resultados = [resultados; time(inicio), time(end), voltage(end)];
end

n_filas_rf = size(rf, 1);
n_filas_resultados = size(resultados, 1);

if n_filas_resultados < n_filas_rf
    resultados = [resultados; zeros(n_filas_rf - n_filas_resultados, 3)];
elseif n_filas_resultados > n_filas_rf
    resultados = resultados(1:n_filas_rf, :);
end

for i = 1:size(resultados, 1)
    t_inicio = resultados(i, 1);
    fila_idx = find(rf(:, 4) == t_inicio, 1);
    if ~isempty(fila_idx)
        rf(fila_idx, 7:9) = resultados(i, :);
    end
end

for i = 1:size(resultados, 1)
    t_final = resultados(i, 2);
    fila_idx = find(rf(:, 5) == t_final, 1);
    if ~isempty(fila_idx)
        rf(fila_idx, 7:9) = resultados(i, :);
    end
end

for i = 1:size(rf, 1)
    t_start = rf(i, 4);
    t_end = rf(i, 5);
    [~, idx_start] = min(abs(time - t_start));
    [~, idx_end] = min(abs(time - t_end));
    v_start = voltage(idx_start);
    v_end = voltage(idx_end);
    rf(i, 2) = sign(v_end - v_start) * abs(rf(i, 2));
end

rf(:,10) = (rf(:,8) - rf(:,7)) / 0.5;
rf(:,11) = rf(:,2) ./ (0.5 * (rf(:,6) - rf(:,10)));

media_senal = mean(voltage);
media_amplitud = mean(abs(rf(:,2)));
media_periodo = mean(rf(:,6));
col11 = rf(:,11);
positivos = col11(col11 > 0);
negativos = col11(col11 < 0);
media_velocidad_positivos = mean(positivos);
media_velocidad_negativos = mean(negativos);

dynamism1 = [media_senal, media_amplitud, media_periodo, media_velocidad_positivos, media_velocidad_negativos];


mediana_amplitud = median(abs(rf(:,2)));
mediana_periodo = median(rf(:,6));
mediana_velocidad_positivos = median(positivos);
mediana_velocidad_negativos = median(negativos);

dynamism2 = [media_senal, mediana_amplitud, mediana_periodo, mediana_velocidad_positivos, mediana_velocidad_negativos];

mode_amplitud = mode(abs(rf(:,2)));
mode_periodo = mode(rf(:,6));
mode_velocidad_positivos = mode(positivos);
mode_velocidad_negativos = mode(negativos);

dynamism3 = [media_senal, mode_amplitud, mode_periodo, mode_velocidad_positivos, mode_velocidad_negativos];

end