from ROOT import TFile, TProfile2D,TCanvas, TGraph2D, TH2F, TH1F, TF1

def find_range(val):
    for i in range(60):
        if (val>(i)) and (val<((i+1))):
            return i
    return None

def make_hist(tree):
    hist = TH2F('hist','hist',60,0,60,60,0,60)
    reshist = TH2F('reshist','reshist',60,0,60,60,0,60)
    myhist = TH2F('myhist','myhist',60,0,60,60,0,60)
    myreshist = TH2F('myreshist','myreshist',60,0,60,60,0,60)
    histlist = {}
    myhistlist = {}
    print 'preparing hists'
    for i in range(60):
        for j in range(60):
            histlist['{}_{}'.format(i,j)] = TH1F('hist_{a}_{b}'.format(a=i,b=j),
                                                 'hist_ecal{a}_hcal{b}'.format(a=i,b=j),20,0,2)
            myhistlist['{}_{}'.format(i,j)] = TH1F('myhist_{a}_{b}'.format(a=i,b=j),
                                                 'myhist_ecal{a}_hcal{b}'.format(a=i,b=j),20,0,2)
    print 'reading tree'
    Nentry = tree.GetEntries()
    k = 0
    for event in tree:
        k+=1
        if k%10000==0:
            print 'event',k,'/',Nentry
        if event.stdEcalib>434.55 and event.stdEcalib<434.56:
            continue
        i = find_range(event.ecal)
        j = find_range(event.hcal)
        if (i is None) or (j is None) :
            continue
        histlist['{}_{}'.format(i,j)].Fill(event.stdEcalib/event.true)
        myhistlist['{}_{}'.format(i,j)].Fill(event.myEcalib/event.true)

    print 'start fitting'
    gauss = TF1('gauss','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
    for i in range(60):
        for j in range(60):
            print 'fitting',i,'/60',j,'/60'
            if histlist['{}_{}'.format(i,j)].Integral()>0.:
                gauss.SetParameters(1,1,0.5)
                histlist['{}_{}'.format(i,j)].Fit(gauss)
                hist.SetBinContent(i+1,j+1,gauss.GetParameter(1))
                hist.SetBinError(i+1,j+1,gauss.GetParError(1))
                reshist.SetBinContent(i+1,j+1,abs(gauss.GetParameter(2)))
                reshist.SetBinError(i+1,j+1,gauss.GetParError(2))
            else:
                hist.SetBinContent(i+1,j+1,0)
                reshist.SetBinContent(i+1,j+1,0)
            if myhistlist['{}_{}'.format(i,j)].Integral()>0.:
                gauss.SetParameters(1,1,0.5)
                myhistlist['{}_{}'.format(i,j)].Fit(gauss)
                myhist.SetBinContent(i+1,j+1,gauss.GetParameter(1))
                myhist.SetBinError(i+1,j+1,gauss.GetParError(1))
                myreshist.SetBinContent(i+1,j+1,abs(gauss.GetParameter(2)))
                myreshist.SetBinError(i+1,j+1,gauss.GetParError(2))
            else:
                myhist.SetBinContent(i+1,j+1,0)
                myreshist.SetBinContent(i+1,j+1,0)
    hist.GetZaxis().SetRangeUser(0.8,1.2)
    myhist.GetZaxis().SetRangeUser(0.8,1.2)
    hist.SetStats(0)
    myhist.SetStats(0)
    hist.GetXaxis().SetTitle('E_{ecal}')
    hist.GetYaxis().SetTitle('E_{hcal}')
    myhist.GetXaxis().SetTitle('E_{ecal}')
    myhist.GetYaxis().SetTitle('E_{hcal}')
    hist.SetTitle('gaussian mean of (standard Ecalib/Etrue)')
    myhist.SetTitle('gaussian mean of (KNN Ecalib/Etrue)')
    
    reshist.GetZaxis().SetRangeUser(0.,0.5)
    myreshist.GetZaxis().SetRangeUser(0.,0.5)
    reshist.SetStats(0)
    myreshist.SetStats(0)
    reshist.GetXaxis().SetTitle('E_{ecal}')
    reshist.GetYaxis().SetTitle('E_{hcal}')
    myreshist.GetXaxis().SetTitle('E_{ecal}')
    myreshist.GetYaxis().SetTitle('E_{hcal}')
    reshist.SetTitle('gaussian standard deviation of (standard Ecalib/Etrue)')
    myreshist.SetTitle('gaussian standard deviation of (KNN Ecalib/Etrue)')

    return hist, myhist, histlist, myhistlist, reshist, myreshist






fil = TFile('all_calibs.root')
tree = fil.Get('s')
hist, myhist, histlist, myhistlist, reshist, myreshist = make_hist(tree)

can1 = TCanvas()
hist.Draw('colz')
can1.SaveAs('gauss_mean_std_zoom.pdf')
can2 = TCanvas()
myhist.Draw('colz')
can2.SaveAs('gauss_mean_KNN_zoom.pdf')

can3 = TCanvas()
reshist.Draw('colz')
can3.SaveAs('gauss_dev_std_zoom.pdf')
can4 = TCanvas()
myreshist.Draw('colz')
can4.SaveAs('gauss_dev_KNN_zoom.pdf')

# tprof = TProfile2D('tprof','profile of Ecalib versus Eecal and Ehcal',100,0,60,50,0,60,0.5,1.5)
# stprof = TProfile2D('stprof','profile of Ecalib versus Eecal and Ehcal',100,0,60,50,0,300,0.5,1.5)

# gra = TGraph2D()

# for i in range(60):
#     for j in range(60):
        

# # print 'nentries', tree.GetEntries()
# # i = 0
# for event in tree:
#     if event.stdEcalib>434.55 and event.stdEcalib<434.56:
#         continue
#     # print event.ecal,event.hcal,event.myEcalib/event.true
#     tprof.Fill(event.ecal,event.hcal,event.myEcalib/event.true)
#     stprof.Fill(event.ecal,event.hcal,event.stdEcalib/event.true)
#     gra.SetPoint(i,event.ecal,event.hcal,event.stdEcalib/event.true)
# #     i+=1
    
# # can = TCanvas()
# # tprof.Draw('COLZ')

# # can2 = TCanvas()
# # stprof.Draw('COLZ')

# # can3 = TCanvas()

# hist = TH2F('hist','hist',60,0,300,60,0,300)
# histlist = []
# for i in range(60):
#     for j in range(60):
#         print 'hist', i , j
#         if tree.GetEntries()>10:
#             hist1 = TH1F('hist_{a}_{b}'.format(a=i*5,b=j*5),
#                         'hist_ecal{a}_hcal{b}'.format(a=i*5,b=j*5),20,0,2)
#             tree.Project('hist_{a}_{b}'.format(a=i*5,b=j*5),'stdEcalib/true',
#                          '(stdEcalib<434.55 || stdEcalib>434.56) && ecal>{ecalinf} && ecal<{ecalsup} && hcal>{hcalinf} && hcal<{hcalsup}'.format(ecalinf=i*5,
#                                                                                                                                                  ecalsup=(i+1)*5,
#                                                                                                                                                  hcalinf=j*5,
#                                                                                                                                                  hcalsup=(j+1)*5))
#             gauss = TF1('gauss','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
#             gauss.SetParameters(1,1,0.5)
#             hist1.Fit(gauss)
#             print 'val =', gauss.GetParameter(1)
#             hist.SetBinContent(i+1,j+1,gauss.GetParameter(1))
#             hist.SetBinError(i+1,j+1,gauss.GetParError(1))

# myhist = TH2F('myhist','myhist',60,0,300,60,0,300)
# myhistlist = []
# for i in range(60):
#     for j in range(60):
#         print 'myhist', i , j
#         if tree.GetEntries()>10:
#             myhist1 = TH1F('myhist_{a}_{b}'.format(a=i*5,b=j*5),
#                         'myhist_ecal{a}_hcal{b}'.format(a=i*5,b=j*5),20,0,2)
#             tree.Project('myhist_{a}_{b}'.format(a=i*5,b=j*5),'myEcalib/true',
#                          '(stdEcalib<434.55 || stdEcalib>434.56) && ecal>{ecalinf} && ecal<{ecalsup} && hcal>{hcalinf} && hcal<{hcalsup}'.format(ecalinf=i*5,
#                                                                                                                                                  ecalsup=(i+1)*5,
#                                                                                                                                                  hcalinf=j*5,
#                                                                                                                                                  hcalsup=(j+1)*5))
#             gauss = TF1('gauss','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
#             gauss.SetParameters(1,1,0.5)
#             myhist1.Fit(gauss)
#             print 'val =', gauss.GetParameter(1)
#             myhist.SetBinContent(i+1,j+1,gauss.GetParameter(1))
#             myhist.SetBinError(i+1,j+1,gauss.GetParError(1))
