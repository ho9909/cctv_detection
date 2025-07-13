import requests
import pprint
import xmltodict
import webbrowser

# Api_Link : https:www//.its.go.kr/opendata/opendataList?service=cctv

def SelectCCTV(maxX, maxY, minX, minY):
    key = "51e23bcd08394488868f5337b96b272c"
    URL = "https://openapi.its.go.kr:9443/cctvInfo"

    # 테스트 값
    # queryParams ="?apiKey=" + key \
    #               + "&type=" + "all" \
    #               + "&cctvType=" + "1" \
    #               + "&minX=" + "126.800000" \
    #               + "&maxX=" + "127.890000" \
    #               + "&minY=" + "34.900000" \
    #               + "&maxY=" + "35.100000" \
    #               + "&getType=" + "xml"

    queryParams ="?apiKey=" + key \
                  + "&type=" + "all" \
                  + "&cctvType=" + "1" \
                  + "&minX=" + minX \
                  + "&maxX=" + maxX \
                  + "&minY=" + minY \
                  + "&maxY=" + maxY \
                  + "&getType=" + "xml"

    URL = URL + queryParams

    response = requests.get(URL)
    contents = response.text
    dict = xmltodict.parse(contents)


    count = 1
    for i in dict['response']['data']:
        print("[", count, "]", "cctvname :", i['cctvname'], "coordx :", i['coordx'], "coordy :", i['coordy'], "cctvurl :", i['cctvurl'])
        count+=1

    select = int(input("입력 :"))
    i = dict['response']['data'][select-1]
    # print(print("[", select, "]", "cctvname :", i['cctvname'], "coordx :", i['coordx'], "coordy :", i['coordy'], "cctvurl :", i['cctvurl']))

    return i['coordx'], i['coordy'], i['cctvurl']


def DetectCCTV(coordx, coordy):     # coordx = 경도 좌표,  coordy = 위도 좌표
    key = "51e23bcd08394488868f5337b96b272c"
    URL = "https://openapi.its.go.kr:9443/cctvInfo"

    err = 0.01
    minX=str(float(coordx)-err)
    maxX=str(float(coordx)+err)
    minY=str(float(coordy)-err)
    maxY=str(float(coordy)+err)

    queryParams = "?apiKey=" + key \
                  + "&type=" + "all" \
                  + "&cctvType=" + "1" \
                  + "&minX=" + minX \
                  + "&maxX=" + maxX \
                  + "&minY=" + minY \
                  + "&maxY=" + maxY \
                  + "&getType=" + "xml"

    URL = URL + queryParams

    response = requests.get(URL)
    contents = response.text
    dict = xmltodict.parse(contents)

    for i in dict['response']['data']:
        print("cctvname :", i['cctvname'], "coordx :", i['coordx'], "coordy :", i['coordy'], "cctvurl :", i['cctvurl'])


a, b, c = SelectCCTV("127.890000", "35.100000", "126.800000", "34.900000")
print(a, b, c)

DetectCCTV(a, b)

# webbrowser.register('chrome', None)
# webbrowser.open("https://naver.com")