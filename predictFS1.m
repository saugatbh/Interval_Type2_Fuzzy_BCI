function [classes]=predictFS1(obj,F)
% This function predicts the classes using the IT2FS classifier for each
% row (observations) of the matrix F i.e. F is the test dataset. This
% function should be used in conjunction with trainFS1.m.
% Created by Saugat Bhattacharyya. Last updated on 28/11/2013.
% Syntax: classes = predictFS1(obj, F)
% Input:
% 1. obj: It is a structure representing the IT2FS classifier (trained).
% This structure is returned by trainFS1.
% 2. F: It is a matrix, whose rows represents the test observations and
% columns represent the features. Let it be of dimension N x d.
% Output:
% 1. classes: It is a column vector of length N, where each element
% indicattes the class of the corresponding observation in F as predicted
% by the IT2FS classifier.
for i=1:obj.nClasses
    for j=1:obj.nFeatures
        % create a row vector x which has #obj.spanMF uniformly 
        % distributed values between the lowest and the highest value 
        % observed along the j-th feature dimension 
        x=linspace(floor(min(obj.f(:,j))),ceil(max(obj.f(:,j))),obj.spanMF);
        % chose the UMF and LMF values corresponding to the i-th class and 
        % j-th feature dimension, then permute the dimension such that Y1
        % and Y2 are row vectors matching the dimension of x.
        Y1=permute(obj.UMF(i,j,1:obj.spanMF),[1 3 2]);
        Y2=permute(obj.LMF(i,j,1:obj.spanMF),[1 3 2]);
        % for the i-th class, use the linear interpolation function to 
        % predict the UMF and LMF values corresponding to the observed 
        % feature value in the test-set F along the j-th feature
        % dimension, if the observed value lies outside the range of x, 
        % extrapolation has been used 
        PredU(:,i,j)=interp1(x,Y1,F(:,j),'linear','extrap');
        PredL(:,i,j)=interp1(x,Y2,F(:,j),'linear','extrap');
    end
end
% PredU and PredL are 3-D matrices of the order #obs x #classes x #feat

for i=1:size(F,1)
    for j=1:obj.nClasses
        % for predicted UMF and LMF, take t-norm or min (joint 
        % contribution) along the feature dimension for each of the 
        % i-th observation in F and for each of the j-th class 
        a(i,j)=min(PredU(i,j,:));
        b(i,j)=min(PredL(i,j,:));
        % for IT2FS, the centroid between LMF and UMF can be considered as
        % the arithmetic mean between the LMF and UMF value 
        Avg(i,j)=(a(i,j)+b(i,j))/2;
    end
end
% Avg is a 2-D matrix of the order #obs x #classes where each element is
% the firing strength of the j-th class by the i-th observation

for i=1:size(F,1)
    % find the maximum firing strength amongst all the classes and store
    % the corresponding class in the loc vector for each of the i-th
    % observation
    loc(i)=1;
    maxAvg=Avg(i,1);
    for j=2:obj.nClasses
        if Avg(i,j)>maxAvg
            loc(i)=j;
            maxAvg=Avg(i,j);
        end
    end
    % find the corresponding class label and store it in the i-th location
    % of the classes vector to be returned
    classes(i,:)=obj.Classes(loc(i));
end
end
