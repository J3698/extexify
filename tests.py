
"""
lx, ly = 0, 0
item = next(iter(val_loader))
unp, lens = pad_packed_sequence(item[0], True)
i = 10
print(labels_dict[item[1][i].item()])

inp = {ort_session.get_inputs()[0].name: unp[i, None, :lens[i], :].numpy()}
out = ort_session.run(None, inp)[0][0]
print(out)
outI = np.argpartition(-out, 5)
print(out[outI][:10])
print([labels_dict[out] for out in outI[:5]])

lp = 1
for pt in unp[i, :lens[i], :]:
    x, y, l = pt.detach().numpy()
    x = 25 + int(x * 450)
    y = 25 + int(y * 450)
    if lp != 1:
        self.a.create_line(lx, ly, x, y)
    print(lx, ly, x, y)
    lx, ly = x, y
    lp = l
"""

"""
_, val_loader, _ = dataloaders(20)
c, total = 0, 0
cc, tt = 0, 0
for x, y in val_loader:
    out = model(x)
    cc += topk_correct(out, y, 5)
    tt += len(y)

    xunp, xlen = pad_packed_sequence(x, True)
    for idx, (xp, xl, yp) in enumerate(zip(xunp, xlen, y)):
        x = xp[None, :xl, :]
        inp = {ort_session.get_inputs()[0].name: x.numpy()}
        o = ort_session.run(None, inp)[0][0]
        print(out[idx][:4].detach(), o[:4])
        outI = np.argpartition(-o, 5)
        c5 = int(yp.item() in outI[:5])
        print(outI[:5], yp)
        c += c5
        total += 1


    print(c / total)
    print(cc / tt)
"""
"""
import pickle
labels_dict = pickle.load(open("./labelsdict.pkl", "rb"))
labels_dict = {i:j for j, i in labels_dict.items()}
class Test():
    def __init__(self):
        self.root = Tk()
        self.a = Canvas(self.root, width = 500, height = 500)
        self.a.pack()

        self.a.bind('<B1-Motion>', self.draw)
        self.a.bind('<ButtonRelease-1>', self.up)
        self.pts = []
        self.root.mainloop()


    def draw(self, xy):
        if torch.rand([]) > 0.3:
            if len(self.pts) != 0 and self.pts[-1][-1] != 1:
                self.a.create_line(self.pts[-1][0], self.pts[-1][1], xy.x, xy.y)
            self.pts.append([xy.x, xy.y, 0])


    def up(self, xy):
        self.pts[-1][-1] = 1
        inp = np.array(self.pts).astype(np.float32)
        inp[:, :2] -= inp[:, :2].min(axis = 0)
        inp[:, :2] /= inp[:, :2].max(axis = 0) + 1e-15
        inp = inp.reshape(1, -1, 3)[:, ::2, :]
        inp = {ort_session.get_inputs()[0].name: inp}
        out = ort_session.run(None, inp)[0][0]
        outI = np.argpartition(-out, 5)
        print([labels_dict[out] for out in outI[:5]])


a = Test()
"""
