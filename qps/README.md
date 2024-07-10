# QR Printing Service
### Target features
- Connects with Dymo LabelWriter 550 (libusb)
- Converts plaintext to QR code bitmap (libqrencode)
- Acts as a simple TCP server listening on the loopback interface & providing easy way to print a label using RID (reagent identifier), as well as enabling the client to be notified when the printer job state changes.