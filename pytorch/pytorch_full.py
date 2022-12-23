import torch
#Autograd(자동미분)

a = torch.randn(3,3)
a = a * 3
print(a)
print(a.requires_grad)

a.requires_grad_(True)

b = (a * a).sum()
print(b)
print(b.grad_fn)

x = torch.ones(3,3, requires_grad=True)
print("x: \n",x)

y = x + 5
# grad fn = add
print("y: \n",y)

z = y * y
# grad fn = mul
print("z: \n",z)

out = z.mean()
# grad fn = mean
print("out: \n", out)

out.backward()

print(x.grad)

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y*2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
y.backward(v)
print(x.grad)

with torch.no_grad():
    print("x.requires_grad", (x**2).requires_grad)


a = torch.ones(2,2, requires_grad=True)
print(a)
print(a.data)
print("before grad",a.grad)
print(a.grad_fn)

b = a + 2
c = b ** 2
print(b)
print(c)
out = c.sum()
print("outoutoutotuotut",out)

out.backward()

print(a.data)
print("after grad",a.grad)
print(a.grad_fn)
print("B")
print(b.data)
print(b.grad)
print(b.grad_fn)
print("C")
print(c.data)
print(c.grad)
print(c.grad_fn)
print("out")
print(out.data)
print(out.grad)
print(out.grad_fn)
