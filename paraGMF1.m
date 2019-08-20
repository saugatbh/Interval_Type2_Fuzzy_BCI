function [m,sd]=paraGMF1(f,targets,Classes,nSets,nClasses,nFeatures)
% This function returns the parametrs (mean,sd) for Gaussian MF using 
% no. of days (nSets).
% Created by Saugat Bhattacharyya. Last updated on 27/11/2013.
% Syntax: [m,sd] = paraGMF1(f,targets,Classes,nSets,nClasses,nFeatures)
% Input:
% 1. f: It is a matrix, whose rows represents the observations and
% columns represent the features. Let it be of dimension N x d.
% 2. targets: It is a column vector of length N identifying the class of 
% each observation of X.
% 3. Classes: It is a vector storing distinguished labels from targets.
% 4. nSets: It is a scalar value denoting #sets concatenated to form f.
% 5. nClasses: It is a scalar value mentioning number of classes
% 6. nFeatures: It is the feature dimension i.e. d with respect to f.
% Output:
% 1. m: It is a 3-D matrix of dimension nClasses x nFeatures x nSets which
% stores the mean feature values.
% 2. sd: It is a 3-D matrix of dimension nClasses x nFeatures x nSets which
% stores the standard deviation among feature values.
m=zeros(nClasses,nFeatures,nSets); % Initializing m
sd=zeros(nClasses,nFeatures,nSets); % Initializing sd
s1=size(f,1); % Total number of observations in f
for k=1:nSets
    % Decomposing f to find each set of observations stored in temp and
    % corresponding labels stored in t. NOTE: All sets are assumed to have
    % equal number of observations
    temp=f((k-1)*(s1/nSets)+1:k*(s1/nSets),:);
    t=targets((k-1)*(s1/nSets)+1:k*(s1/nSets),:);
    for i=1:nClasses
        % Obtaining mean and sd for those observations from temp where
        % labels correspond to the i-th class
        m(i,:,k)=mean(temp(t==Classes(i),:));
        sd(i,:,k)=std(temp(t==Classes(i),:));
    end
end
