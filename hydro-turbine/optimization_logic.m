%% Hydro-Turbine Optimization Using Lagrange Multipliers
% Based on the project by Alex Korde, MATH 2310
% Recreated from PDF description

clear; clc; close all;

%% Define constants from the derived equations
% Optimal flow distribution formulas (from Procedure 1)
% Q1 = 0.341*QT - 75.134
% Q2 = 0.297*QT + 20.995
% Q3 = 0.362*QT + 54.285

%% Turbine flow limits (from Procedure 2)
Q1_max = 1110;  % from PDF: Q1 <= 1110? Actually says Q1 <= 3475 but check
Q2_max = 1110;
Q3_max = 1225;  % from PDF: Q3 <= 1225 (or 3231? Using stricter)

% Actually from PDF page 4-5:
% Q1 <= 1110, Q2 <= 1110, Q3 <= 1225 are the operating limits

%% Procedure 3: Given total flow QT, compute optimal distribution
QT_input = 2500; % ft^3/s

Q1_opt = 0.341*QT_input - 75.134;
Q2_opt = 0.297*QT_input + 20.995;
Q3_opt = 0.362*QT_input + 54.285;

% Check limits
if Q1_opt > Q1_max || Q2_opt > Q2_max || Q3_opt > Q3_max
    warning('Flow exceeds turbine limits for QT = %d', QT_input);
end

%% Define power functions (from PDF page 3)
% KW_i = (A_i + B_i*Q_i + C_i*Q_i^2) * (170 - 1.6e-6 * QT^2)
% Note: In PDF, the quadratic coefficients are negative.

% Coefficients for each turbine
coeff = struct();
coeff(1).A = -18.89;
coeff(1).B = 0.1277;
coeff(1).C = -4.08e-5;

coeff(2).A = -24.51;
coeff(2).B = 0.1358;
coeff(2).C = -4.69e-5;

coeff(3).A = -27.02;
coeff(3).B = 0.1380;
coeff(3).C = -3.84e-5;

% Head loss factor (same for all turbines)
head_factor = @(QT) (170 - 1.6e-6 * QT^2);

% Power function for turbine i
KW = @(i, Q, QT) (coeff(i).A + coeff(i).B*Q + coeff(i).C*Q^2) * head_factor(QT);

%% Compute power at optimal flows
QT = QT_input;
KW1_val = KW(1, Q1_opt, QT);
KW2_val = KW(2, Q2_opt, QT);
KW3_val = KW(3, Q3_opt, QT);
total_power = KW1_val + KW2_val + KW3_val;

%% Display results
fprintf('=== Hydro-Turbine Optimization Results ===\n');
fprintf('Total flow QT = %.2f ft^3/s\n\n', QT);
fprintf('Optimal flow distribution:\n');
fprintf('  Q1 = %.2f ft^3/s\n', Q1_opt);
fprintf('  Q2 = %.2f ft^3/s\n', Q2_opt);
fprintf('  Q3 = %.2f ft^3/s\n', Q3_opt);
fprintf('  Sum = %.2f ft^3/s\n', Q1_opt+Q2_opt+Q3_opt);
fprintf('\nPower output:\n');
fprintf('  KW1 = %.2f kW\n', KW1_val);
fprintf('  KW2 = %.2f kW\n', KW2_val);
fprintf('  KW3 = %.2f kW\n', KW3_val);
fprintf('  Total = %.2f kW\n', total_power);

%% Compare with equal distribution (for validation)
Q_eq = QT/3;
KW1_eq = KW(1, Q_eq, QT);
KW2_eq = KW(2, Q_eq, QT);
KW3_eq = KW(3, Q_eq, QT);
total_eq = KW1_eq + KW2_eq + KW3_eq;

fprintf('\n=== Comparison with Equal Distribution ===\n');
fprintf('Equal Q = %.2f each\n', Q_eq);
fprintf('Total power (equal) = %.2f kW\n', total_eq);
fprintf('Improvement = %.2f kW (%.2f%%)\n', total_power - total_eq, ...
    (total_power - total_eq)/total_eq * 100);

%% Procedure 4: Test for QT = 1000 and QT = 600
QT_test = [1000, 600];
for k = 1:length(QT_test)
    QT_val = QT_test(k);
    Q1v = 0.341*QT_val - 75.134;
    Q2v = 0.297*QT_val + 20.995;
    Q3v = 0.362*QT_val + 54.285;
    KW1v = KW(1, Q1v, QT_val);
    KW2v = KW(2, Q2v, QT_val);
    KW3v = KW(3, Q3v, QT_val);
    totalv = KW1v + KW2v + KW3v;
    
    fprintf('\n=== QT = %.0f ft^3/s ===\n', QT_val);
    fprintf('Q1=%.2f, Q2=%.2f, Q3=%.2f\n', Q1v, Q2v, Q3v);
    fprintf('Total power = %.2f kW\n', totalv);
end

%% Procedure 5: Two-turbine optimization for QT=1500
% From PDF: Q2 = 86.354 + 0.870*Q1  (for turbines 1&2)
% For turbines 1&3: Q3 = 134.115 + 1.063*Q1
% For turbines 2&3: Q3 = 28.646 + 1.221*Q2

QT_2turb = 1500;

% Case 1: Turbines 1 and 2
% Q1 + Q2 = QT, Q2 = 86.354 + 0.870*Q1
% Solve: Q1 + (86.354 + 0.870*Q1) = QT
Q1_12 = (QT_2turb - 86.354) / (1 + 0.870);
Q2_12 = QT_2turb - Q1_12;
KW1_12 = KW(1, Q1_12, QT_2turb);
KW2_12 = KW(2, Q2_12, QT_2turb);
total_12 = KW1_12 + KW2_12;

% Case 2: Turbines 1 and 3
% Q3 = 134.115 + 1.063*Q1, Q1 + Q3 = QT
Q1_13 = (QT_2turb - 134.115) / (1 + 1.063);
Q3_13 = QT_2turb - Q1_13;
KW1_13 = KW(1, Q1_13, QT_2turb);
KW3_13 = KW(3, Q3_13, QT_2turb);
total_13 = KW1_13 + KW3_13;

% Case 3: Turbines 2 and 3
% Q3 = 28.646 + 1.221*Q2, Q2 + Q3 = QT
Q2_23 = (QT_2turb - 28.646) / (1 + 1.221);
Q3_23 = QT_2turb - Q2_23;
KW2_23 = KW(2, Q2_23, QT_2turb);
KW3_23 = KW(3, Q3_23, QT_2turb);
total_23 = KW2_23 + KW3_23;

fprintf('\n=== Two-Turbine Optimization (QT = %.0f) ===\n', QT_2turb);
fprintf('Turbines 1&2: Q1=%.2f, Q2=%.2f → Total = %.2f kW\n', Q1_12, Q2_12, total_12);
fprintf('Turbines 1&3: Q1=%.2f, Q3=%.2f → Total = %.2f kW\n', Q1_13, Q3_13, total_13);
fprintf('Turbines 2&3: Q2=%.2f, Q3=%.2f → Total = %.2f kW\n', Q2_23, Q3_23, total_23);
[best_total, best_idx] = max([total_12, total_13, total_23]);
fprintf('Best combination: Turbines %d&%d with %.2f kW\n', ...
    best_idx==1&&1||2, best_idx==2&&1||3, best_idx==3&&2||3, best_total);

%% Plot power vs total flow for all three turbines (Figure similar to PDF)
QT_range = 500:10:3000;
Q1_range = 0.341*QT_range - 75.134;
Q2_range = 0.297*QT_range + 20.995;
Q3_range = 0.362*QT_range + 54.285;

% Ensure non-negative flows
Q1_range(Q1_range < 0) = 0;
Q2_range(Q2_range < 0) = 0;
Q3_range(Q3_range < 0) = 0;

KW1_range = arrayfun(@(q, qt) KW(1, q, qt), Q1_range, QT_range);
KW2_range = arrayfun(@(q, qt) KW(2, q, qt), Q2_range, QT_range);
KW3_range = arrayfun(@(q, qt) KW(3, q, qt), Q3_range, QT_range);

figure;
plot(QT_range, KW1_range, 'r-', 'LineWidth', 2); hold on;
plot(QT_range, KW2_range, 'g-', 'LineWidth', 2);
plot(QT_range, KW3_range, 'b-', 'LineWidth', 2);
xlabel('Total Flow Q_T (ft^3/s)');
ylabel('Power (kW)');
title('Turbine Power vs. Total Flow at Optimal Distribution');
legend('Turbine 1', 'Turbine 2', 'Turbine 3');
grid on;

% Mark the point at QT=600 as in PDF
xline(600, '--k', 'QT=600');
fprintf('\nPlot generated: Power vs. QT for all three turbines.\n');