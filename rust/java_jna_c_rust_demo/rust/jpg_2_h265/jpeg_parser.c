#include <jpeg_parser.h>

void JPEG_ParaseSOF(JPEGINFO_t *ptJPEGINFO, const UCHAR *pucStream)
{
    /*---------------------------------------------------------------------|
      |  [0]  |  [1]  |  [2]  |  [3]  |  [4]  |  [5]  |  [6]  |  [7]  | ...|
      |  Chuck Len.   |Preces.|    Height     |     Width     |No.Comp| ...|
      ---------------------------------------------------------------------|*/
    //Reference. ../doc/JPEG_ParserSOF.jpg

    //UINT uiWidth = 0, uiHeight = 0, uiNumOfColorComponents=0;

    ptJPEGINFO->uiHeight = BYTEtoWORD(pucStream+3);
    ptJPEGINFO->uiWidth  = BYTEtoWORD(pucStream+5);
    
    ptJPEGINFO->uiColorComponents=(pucStream[7]);

    /*printf("Image Information:\nWidth=%d, Height=%d\nColorComponents=%d\n",
           uiWidth, uiHeight, uiColorComponents);*/
}

INT JPEG_MarkerParse(JPEGINFO_t *ptJPEGINFO, const UCHAR *pucStream,
                     UINT *puiStreamSize)
{
    /*uiMarkerSOS was checked marker SOS, than end mark parser
      uiCheckFormat was checked JPEG format again
      uiChuckLen is lenth of this maker to offset next marker*/
    UINT uiMarkerSOS = 0, uiCheckFormat = 1, uiChuckLen;

    //Temp for maker data    
    INT iMaker;
    
    //Chek uiMarkerSOS, uiCheckFormat and uipStreamSize
    while (!uiMarkerSOS && uiCheckFormat && puiStreamSize > 0)
    {
        //Not really JPEG format
        if (*pucStream != 0xFF)
        {
            uiCheckFormat=0;
        }
        
        //Find the mark data after 0xFF
        while (*pucStream == 0xFF && puiStreamSize >0)
        {
            pucStream++;

            (*puiStreamSize)--;    
        }

        //Get marker date and offset to next byte for ChuckLen
        iMaker = *pucStream++;

        (*puiStreamSize)--;

        //Get ChuckLen
        uiChuckLen = BYTEtoWORD(pucStream);

        //Parser maker
        switch (iMaker)
        {
            case SOF:
                //Parser marker of SOF
                JPEG_ParaseSOF(ptJPEGINFO,pucStream);
                break;

            case SOS:
                //End parser marker
                uiMarkerSOS = 1;
                break;
        }
        
        //Offset to next mark position
        pucStream = pucStream + uiChuckLen;
    }

    //Check not really JPEG format
    if (!uiCheckFormat || puiStreamSize == 0)
    {
        printf("It's not really JPEG format\n");
        return -1;
    }

    return 0;
}

INT JPEG_HeaderParser(JPEGINFO_t *ptJPEGINFO , UCHAR *pucFileBuffer,  UINT puiFileBufferSize)

{
    //Locat to stream after being signal(0xFFD8)
    const UCHAR *pucStartStream = pucFileBuffer + 2;

    puiFileBufferSize = puiFileBufferSize - 2;

    //Check the file is JPEG format or not
    if (pucFileBuffer[0] != 0xFF || pucFileBuffer[1] != SOI)
    {
        printf("It's not JPEG file\n");
        
        return -1;
    }

    //Start header's marker parser
    if (JPEG_MarkerParse(ptJPEGINFO, pucStartStream, &puiFileBufferSize) < 0)
    {
        return -1;
    }

    return 0;
}
