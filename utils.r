source('~/thib/projects/tools/R_lib.r')
setwd('~/thib/projects/reliable_info')
### LIST OF FUNCIONS
## extract_group(fit.name, parameters
## extract_indiv(fit.name, parameters
## generate_pred(fit.name)
## simul_stim(T,I, proba_color, info_values, proba_info)
## simul_llo(alpha, beta, color_info, proba_info)
## simul_linear(alpha, beta, color_info, proba_info)



##############################################
#  EXTRACT  GROUP PARAMETERS
#############################################
# fit.name = fit output file
# parameters = list of parameters to be extracted
# return a dataframe results.group

extract_group <- function(fit.name, parameters){
    fit.name <- fit.name
    parameters <- parameters
    fit <- readRDS(fit.name)
    mat = matrix(ncol = 0, nrow = 0)
    su = data.frame(mat)
    ## log_lik/AIC
    log_lik = loo::extract_log_lik(fit, parameter_name = "log_lik", merge_chains = TRUE)
    l <- loo::elpd(log_lik)$estimates[[1]]
    AIC <- -2*l + 2*length(parameters)
    ## Divergent
    sampler_params <- get_sampler_params(fit, inc_warmup = FALSE)
    div = sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))
    ## Parameters
    results.group <- as.data.frame(summary(fit, pars = parameters, include = TRUE,  probs = c(0.025,  0.50,  0.975), digits = 3)$summary) %>%
        rownames_to_column() %>%
        rename(param = rowname) %>%
        mutate(AIC = AIC, l = l)
    return (results.group)
}

#############################################
# EXTRACT INDIVIDUAL PARAMETERS
###########################################
# fit.name = fit output file
# parameters = list of parameters to be extracted
# return a dataframe results.group

extract_individual <- function(fit.name, parameters){
    fit.name <- fit.name
    parameters <- parameters
    mat = matrix(ncol = 0, nrow = 0)
    su = data.frame(mat)
    fit <- readRDS(fit.name)
    su_temp <- as.data.frame(summary(fit, pars = parameters, include = TRUE,  probs = c(0.025,  0.50,  0.975), digits = 3)$summary) %>%
        rownames_to_column() %>%
        rename(param = rowname) 
    results.individual <- su_temp %>%
        mutate(s  =  as.numeric(gsub("[^0-9]", "", param))) %>%
        mutate(parameter = gsub("[^A-Za-z]", "", param)) %>%
    return(results.individual)
}

###########################################
#  GENERATE PREDICTION
##########################################
# generate prediction from fitted model
# fit.name: fit output file
# output: pred

## Small function to compute the mode
comp_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

generate_pred <- function(fit.name){
    fit.name <- fit.name
    pred_df = data.frame(subject = numeric(), trial = numeric(), pred = numeric())
    fit <- readRDS(fit.name)
    pred <- extract(fit)$y_pred  ## (n_chains x Chains_length)x(n_subj)x(n_trials)
    n_subj = dim(pred)[2]
    for (i in c(1:n_subj)){
        pred_temp <- extract(fit)$y_pred[,i,] 
        pred_temp <-  as.data.frame(pred_temp) %>% 
            summarize_all(comp_mode) %>%
            t() %>%
            as.data.frame() %>%
            mutate(trial =row_number()) %>%
            rename(pred = V1) %>%
            mutate(subject = i) %>%
            mutate(s = i) %>% ## subject index
            mutate(trial = row_number())                
        pred_df <- rbind(pred_df, pred_temp)
    }
    return(pred_df)
}

###############################################
#     SIMULATE STIMULI
##############################################

## T = number of trials
## I = number of info samples
## proba_color = probability of Blue vs Right
## info_values = possible proba values for the info (ex: (.5,.55,.65))
## proba_info = proba of each info value (ex: (1/3,1/3,1/3))
## return list(color_array, proba_array)

simul_stim <- function(T,I, proba_color, info_values, proba_info){
    color_array <-  array(rep(0,T*I), dim=c(T,I))
    proba_array <-  array(rep(0, T*I), dim=c(T,I))
    color_true <- 1+rbern(T,proba_color)
    for (t in c(1:T)){
        for (i in c(1:I)){
            p = sample(info_values, size = 1, prob = proba_info)
            proba_array[t,i] = p
            if (rbern(1,p) == 1){
                color_array[t,i] = color_true[t]
            }else{
                color_array[t,i] = -color_true[t]+3
            }
        }
    }
    return(list(color_array, proba_array))
}

###############################################
#     SIMULATE LOGLIKELIHOOD RESPONSE
##############################################
## alpha, beta: parameters of the likelihood transformation
## color_info: TxI array : info color for each trial (row) and info sample (column)
## proba_info: TxI array : info proba for each trial (row) and info sample (column)
## weights : I-vector
simul_log <- function(alpha, beta, weights, color_info, proba_info){
    ## initialisation
    T = nrow(color_info)
    I = ncol(color_info)
    choice  =rep(-1,T)
    evidence_1  =rep(-1,T)
    evidence_2  =rep(-1,T)
    ## generate responses
    for (t in c(1:T)){
        evidence = c(0,0) ## initial value
        for (i in c(1:I)){
            if (color_info[t,i] == 1){## blue
                evidence[1] = evidence[1]+weights[i]*(alpha*log(proba_info[t,i]/(1-proba_info[t,i]))+beta)          
            }else{## red 
                evidence[2] = evidence[2]+weights[i]*(alpha*log(proba_info[t,i]/(1-proba_info[t,i]))+beta) 
            }
        }
        x <- plogis(evidence[2] - evidence[1])
        choice[t] <- rbern(1, x) + 1
        evidence_1[t] = evidence[1]
        evidence_2[t] = evidence[2]
    }
    return(list(choice,evidence_1,evidence_2))
}

###############################################
#     SIMULATE LINEAR RESPONSE
##############################################
## simulate choices
## alpha, beta: parameters of the linear transformation
## color_info: TxI array : info color for each trial (row) and info sample (column)
## proba_info: TxI array : info proba for each trial (row) and info sample (column)

simul_linear <- function(alpha, beta, weights, color_info, proba_info){
    ## initialisation
    T = nrow(color_info)
    I = ncol(color_info)
    choice  =rep(-1,T)
    evidence_1  =rep(-1,T)
    evidence_2  =rep(-1,T)
    ## generate responses
    for (t in c(1:T)){
    evidence = c(0,0) ## initial value
        for (i in c(1:I)){
            if (color_info[t,i] == 1){## blue
                evidence[1] = evidence[1] + weights[i]*(alpha*(proba_info[t,i])+beta)          
            }else{## red 
                evidence[2] = evidence[2] + weights[i]*(alpha*(proba_info[t,i])+beta  )
            }
        }
        x <- plogis(evidence[2] - evidence[1])
        choice[t] <- rbern(1, x) + 1
        evidence_1[t] = evidence[1]
        evidence_2[t] = evidence[2]
    }
    return(list(choice,evidence_1,evidence_2))
}
