#ifndef _WINDOW_H_
#define _WINDOW_H_

class WindowFunc {
public:
    const char* Name();
    virtual double* Build(int len) = 0;
    const double* FreqResponse(int len);
    const double* AmpResponse(int len);
    void RenderWidget();

    float CohGain();
    float EqNBW();
    float ProcGain();
    float ScalLoss();
    float WorstProcLoss();
    float OvrCor25();
    float OvrCor50();
    float OvrCor75();
    float AmpFlat25();
    float AmpFlat50();
    float AmpFlat75();
    float MainLobeWidth3();
    float MainLobeWidth6();
    float PeakSideWidth3();
    float PeakSideWidth6();
    float PeakSideLevel();
    //float OptimalOverlap();
    //const double* OverlapAmpRespOpt();
    //const double* OverlapAmpResp25();
    //const double* OverlapAmpResp50();
    //const double* OverlapAmpResp75();

    void Init();

    virtual ~WindowFunc();

    WindowFunc(const WindowFunc&) = delete;
    WindowFunc& operator=(const WindowFunc&) = delete;
protected:
    explicit WindowFunc(const char* name);

private:
    char* name = nullptr;

    float cohGain         = 0;
    bool cohGainSet       = false;
    float eqNBW           = 0;
    bool eqNBWSet         = false;
    float procGain        = 0;
    bool procGainSet      = false;
    float scalLoss        = 0;
    bool scalLossSet      = false;
    float worstProcLoss   = 0;
    bool worstProcLossSet = false;
    float ovrCor25        = 0;
    bool ovrCor25Set      = false;
    float ovrCor50        = 0;
    bool ovrCor50Set      = false;
    float ovrCor75        = 0;
    bool ovrCor75Set      = false;
    float ampFlat25       = 0;
    bool ampFlat25Set     = false;
    float ampFlat50       = 0;
    bool ampFlat50Set     = false;
    float ampFlat75       = 0;
    bool ampFlatSet75     = false;
    float mlw3            = 0;
    bool mlw3Set          = false;
    float mlw6            = 0;
    bool mlw6Set          = false;
    float slw3            = 0;
    bool slw3Set          = false;
    float slw6            = 0;
    bool slw6Set          = false;
    float psl             = 0;
    bool pslSet           = false;
    float optOvr          = 0;
    bool optOvrSet        = false;

    double* freqResponse    = nullptr;
    int     freqResponseLen = 0;
    double* ampResponse     = nullptr;
    int     ampResponseLen  = 0;
    //double* overlapOpt      = nullptr;
    //int     overlapOptLen   = 0;
    //double* overlap25       = nullptr;
    //int     overlap25Len    = 0;
    //double* overlap50       = nullptr;
    //int     overlap50Len    = 0;
    //double* overlap75       = nullptr;
    //int     overlap75Len    = 0;

    float ovrCor(float ovr);
    float ampFlat(float ovr);
    float lobeMeas(double* lobe, int len);
    const double* overlapAmpResp(float ovr, unsigned* len);
};

#endif