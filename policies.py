class MarkowitzPolicy:
    def __init__(self):
        self.portfolios = self.init_portfolios()

    def get_action(self,memory):
        df1 = memory.copy().reset_index(drop=True).tail(30)
        for i,row in df1.iterrows():
            for i2,row2 in self.portfolios.iterrows():
                df1.loc[i,f"p{i2} value"] = np.dot(row,row2)
        for i in range(NPORTS):
            df1[f"p{i} return"] = df1[f"p{i} value"].diff()
        
        df2 = df1.agg(["mean","std"])[[f"p{i} return" for i in range(NPORTS)]].T
        df2.loc[df2["std"].isnull(),"std"]=1e-8
        df2["sharpe"] = df2["mean"]/df2["std"]
        
        
        max_ = df2.sort_values("std").head(df2.shape[0]//4)["mean"].max()
        
        max_ = df2[df2["mean"]==max_].index[0].strip("p").strip(" return")
        max_ = int(max_)
        action = self.portfolios.loc[max_,:].values/100
        action = np.hstack((0,action))
        return action

    def init_portfolios(self):
        if FIXED_PORTFOLIO:
            ps = pd.read_csv("portfolios_sample.csv")
        else:
            portfolios = []
            for i in range(NPORTS):
                num1 = np.random.randint(100)
                num2 = np.random.randint(100-num1)
                num3 = 100-num1-num2
                portfolios.append([num1,num2,num3])

            ps = pd.DataFrame(portfolios,columns=COINS)
            # ps.to_csv("portfolios_sample.csv",index=None)
        return ps    
