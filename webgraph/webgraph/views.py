from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show, colorbar
import matplotlib.colors
from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Create your views here.

def dashboard(request, *args, **kwargs):
    context = {
        'display' : 'none'
    }

    if request.method == "POST":

        try:
            try:
                os.remove(os.path.join(BASE_DIR, 'static/webgraph/ad.png'))
                os.remove(os.path.join(BASE_DIR, 'static/webgraph/rd.png'))
            except:
                pass

            sp = request.POST.get('sp')
            dg = request.POST.get('dg')
            tp = request.POST.get('tp')
            pz = request.POST.get('pz')
            sc = request.POST.get('sc')
            ct = request.POST.get('ct')
            
            sp, dg, tp, pz, sc, ct = float(sp), float(dg), int(tp), float(pz), float(sc), float(ct)

            # -*- coding: utf-8 -*-
            # %% Code cell: Set Parameters
            # PARAMETERS
            diagnosticity = dg/100
            T = tp
            implementation_cost = sc
            testing_cost = ct
            prize = pz
            prior_success = sp/100

            # %% Code cell: Generate vectors needed for calculations
            thresholds_implementation = np.empty((100,100))
            thresholds_implementation[:] = np.nan
            thresholds_termination = np.empty([100, 100])
            thresholds_termination[:] = np.nan

            # %% Code cell: Start loop to calculate stopping behavior for any agent

            column = 0
            while column <=99:
                
                row = 0
                while row <=99:
                    
                    c_underinference = column/100+0.01
                    d_brn = row/100+0.01
                    
                    # Create Vectors
                    Strategy = np.empty((T+T+1,T+1))
                    V = np.empty((T+T+1,T+1)) #V is the binomial tree
                    V[:] = np.nan
                    M = np.empty((T+T+1,7)) #M is vector that contains k (column 1), pSgk (column 2), pFgk (column 3), psgk (column 4), pfgk (column 5), payoff_commercialize (column 6), payoff_stop (column 7)
                    M[:] = np.nan
                    K = np.empty((3,T+1)) #for each t (row 1) K contains implementation threshold (row 2), termination threshold (row 3) 
                    K[:] = np.nan
                    
                    
                    # Calculate Posteriors for each possible k
                    k = T
                    while k >=-T:
                        
                        M[T+k,0] = k
                        pSgk = (((diagnosticity/(1-diagnosticity))**k))**c_underinference /((((1-prior_success)/prior_success))**d_brn+((diagnosticity/(1-diagnosticity))**k)**c_underinference)
                        M[T+k,1] = pSgk
                        pFgk = 1-(((diagnosticity/(1-diagnosticity))**k))**c_underinference /((((1-prior_success)/prior_success))**d_brn+((diagnosticity/(1-diagnosticity))**k)**c_underinference)
                        M[T+k,2] = pFgk
                        psgk = diagnosticity*pSgk+(1-diagnosticity)*pFgk
                        M[T+k,3] = psgk
                        pfgk = diagnosticity*pFgk+(1-diagnosticity)*pSgk
                        M[T+k,4] = pFgk
                        implement = -implementation_cost+pSgk * prize
                        M[T+k,5] = implement
                        stop = max(-implementation_cost+pSgk * prize,0)
                        M[T+k,6]=stop
                        
                        k -= 1
                    
                    # Calculate V_T
                    k = T
                    while k >=-T:
                        

                        V[T+k,T]=M[T+k,6]
                        # Strategy in T (1 if implement and 0 if terminate)
                        if V[T+k,T]==0:
                            Strategy[T+k,T]=0
                            thresholds_termination[row,column]=T
                        elif V[T+k,T]==M[T+k,5]:
                            Strategy[T+k,T]=1
                            thresholds_implementation[row,column]=T
                            
                        k-= 2

                    
                    # Backward Induction V_t(t,k)
                    t=T-1
                    while t>=1:
                        
                        k=t
                        while k>=-t:
                            
                            test = -testing_cost+M[T+k,3]*V[T+k+1,t+1]+M[T+k,4]*V[T+k-1,t+1]
                            value_function = max(M[T+k,6],test)
                            V[T+k,t]=value_function
                            # Strategy in t
                            if Strategy[T+k+1,t+1]==1 and Strategy[T+k-1,t+1]==1:
                                Strategy[T+k,t]=1
                                thresholds_implementation[row,column]=t
                            elif Strategy[T+k+1,t+1]==0 and Strategy[T+k,t+1]==0:
                                Strategy[T+k,t]=0
                                thresholds_termination[row,column]=t
                            elif V[T+k,t]==M[T+k,5]:
                                Strategy[T+k,t]=1
                                thresholds_implementation[row,column]=t
                            elif V[T+k,t]==0:
                                Strategy[T+k,t]=0
                                thresholds_termination[row,column]=t
                            else:
                                Strategy[T+k,t]=2
                            
                            k-=2
                        
                        t-=1
                    
                    # Backward Induction V_0
                    test_start = -testing_cost+(prior_success*diagnosticity+(1-prior_success)*(1-diagnosticity))*V[T+1,1]+(prior_success*(1-diagnosticity)+(1-prior_success)*diagnosticity)*V[T-1,1]
                    value_function_start = max(max(0,-implementation_cost+prior_success * prize),test_start)
                    V[T,0]=value_function_start
                    # Strategy in t
                    if Strategy[T+1,1]==1 and Strategy[T-1,1]==1:
                        Strategy[T,0]=1
                        thresholds_implementation[row,column]=0
                        thresholds_termination[row,column]=0
                    elif Strategy[T+1,1]==0 and Strategy[T-1,1]==0:
                        Strategy[T,0]=0
                        thresholds_implementation[row,column]=0
                        thresholds_termination[row,column]=0
                    elif V[T,0]==(-implementation_cost+prior_success * prize):
                        Strategy[T,0]=1
                        thresholds_implementation[row,column]=0
                        thresholds_termination[row,column]=0
                    elif V[T,0]==0:
                        Strategy[T,0]=0
                        thresholds_implementation[row,column]=0
                        thresholds_termination[row,column]=0
                    else:
                        Strategy[T,0]=2
                                
                    del V
                    del M
                    del K
                    
                    row += 1
                    
                column += 1
            
            # %% Code cell: Generate plots

            plot_thresholds_implementation=np.flipud(thresholds_implementation)
            plot_thresholds_termination=np.flipud(thresholds_termination)

            #Implementation Thresholds
            #To-Do: Improve resolution of plot
            #To-Do: Change colorbar boundaries from min_val to max_val without changing colorscheme and only label discrete values
            matrix_flattened = thresholds_implementation.flatten()
            min_val = min(matrix_flattened)
            max_val = max(matrix_flattened)
            if max_val - thresholds_implementation[99,99] > thresholds_implementation[99,99] - min_val:
                norm=plt.Normalize(thresholds_implementation[99,99]-max_val,max_val)
            else:
                norm=plt.Normalize(min_val, 2*(thresholds_implementation[99,99]-min_val))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","black","red"])
            imshow(plot_thresholds_implementation,extent=[0,1,0,1], cmap=cmap, norm=norm)
            colorbar()
            plt.xlabel('underinference')
            plt.ylabel('base-rate neglect')
            plt.title('Acceptance Domain')
            plt.savefig(os.path.join(BASE_DIR, 'static/webgraph/ad.png'))
            plt.clf()

            #Termination Thresholds
            #To-Do: Improve resolution of plot
            #To-Do: Change colorbar boundaries from min_val to max_val without changing colorscheme and only label discrete values
            matrix_flattened = thresholds_termination.flatten()
            min_val = min(matrix_flattened)
            max_val = max(matrix_flattened)
            if max_val - thresholds_termination[99,99] > thresholds_termination[99,99] - min_val:
                norm=plt.Normalize(thresholds_termination[99,99]-max_val,max_val)
            else:
                norm=plt.Normalize(min_val, 2*(thresholds_termination[99,99]-min_val))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","black","red"])
            imshow(plot_thresholds_termination,extent=[0,1,0,1], cmap=cmap, norm=norm)
            colorbar()
            plt.xlabel('underinference')
            plt.ylabel('base-rate neglect')
            plt.title('Rejection Domain')
            plt.savefig(os.path.join(BASE_DIR, 'static/webgraph/rd.png'))
            plt.clf()
            plt.close()

            context['display'] = "block"
        except:
            pass


    return render(request, 'webgraph/dashboard.html', context)