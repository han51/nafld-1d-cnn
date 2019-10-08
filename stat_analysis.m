%% Matlab code for statistical analysis
%% Classifier part 1
clear all; close all
load classifier_without_tgc.mat     % load classifier results 

labels = y_label_pt;  % true labels for patients;
scores = y_raw_pt;    % predicted socres;
preds = scores > 0.5;  % binary prediction;

cp = classperf(1-labels, 1-preds); % returns classifier performance 
% confidence intervals can be calculated from https://www.medcalc.org/calc/diagnostic_test.php

cp.DiagnosticTable

%% Fat fraction estimator 
clear all; close all
load ff_estimator_without_tgc.mat     % load ff_estimator results

pdff = y_label_pt(:);    % true pdff for patients;
ff = y_raw_pt(:);    % predicted fat fraction;

[r1] = corr(pdff, ff,'Type','Spearman')
[r2] = corr(pdff, ff,'Type','Pearson')

x = pdff;
y = ff;

md1 = fitlm(x,y)

[pdff_sorted, idx] = sort(pdff);
ff_sorted = ff(idx);

for i = length(pdff_sorted):-1:5
    x = pdff_sorted(1:i);
    y = ff_sorted(1:i);
    md3 = fitlm([x x.^2 x.^3], y);
    md2 = fitlm([x x.^2], y);
    md1 = fitlm(x,y);
    p1 = md1.Coefficients.pValue(end);
    p2 = md2.Coefficients.pValue(end);
    p3 = md3.Coefficients.pValue(end);
    if (p3 >= 0.05) & (p2 >= 0.05)
        break
    end
end
pdff(i)
pdff(i+1)
md1



