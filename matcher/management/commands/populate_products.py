import requests
from io import BytesIO
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings
from matcher.models import Product
from matcher.similarity import extract_features
from PIL import Image as PILImage
import numpy as np
import re
import traceback
import sys

class Command(BaseCommand):
    help = 'Populates the database and organizes product images into category folders on S3 using normal names.'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting product population with S3 category folders...")

        if not getattr(settings, 'AWS_STORAGE_BUCKET_NAME', None):
            self.stdout.write(self.style.ERROR(
                "S3 Bucket Name is not configured. Please check your .env file and ensure that "
                "AWS_STORAGE_BUCKET_NAME is set correctly."
            ))
            sys.exit(1)
        
        Product.objects.all().delete()
        self.stdout.write("Cleared existing products.")

        products_data = [
{
            "name": "Round Neck Black T-Shirt",
            "category": "Clothing",
            "description": "Classic round neck black t-shirt for everyday casual wear",
            "image_url": "https://imgs.search.brave.com/MkZDrluUhvM25s4mvpaCuPJQzWhLpJr93f9ctH_S7cY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/am9ja2V5LmluL2Nk/bi9zaG9wL2ZpbGVz/LzI3MTRfQkxBQ0tf/MDEwNV9TMTIzX0pL/WV81XzBhOWE4M2Ix/LWM1NGEtNDIyZi05/NzMzLWE3ODA3NzE2/ZDM3MS53ZWJwP3Y9/MTcyNTYxOTgzNyZ3/aWR0aD00MjA"
        },
        {
            "name": "Women Silver Watch",
            "category": "Accessories",
            "description": "Elegant silver analog watch designed for women",
            "image_url": "https://imgs.search.brave.com/WP8LoQSFIsfa1f9w0g5dmG6VgwhgZzEoE8Pvkm3ZPzU/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NzFaeW9FUVR3REwu/anBn"
        },
        {
            "name": "Men’s Casual Black Shirt",
            "category": "Clothing",
            "description": "Soft and comfortable casual black t-shirt for men",
            "image_url": "https://imgs.search.brave.com/bMoHau955rPAzBfkCG8JMbpYZByJhqJVGiBR8vyZgls/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly94Y2Ru/Lm5leHQuY28udWsv/Q29tbW9uL0l0ZW1z/L0RlZmF1bHQvRGVm/YXVsdC9JdGVtSW1h/Z2VzLzNfNFJhdGlv/L1NlYXJjaC9MZ2Uv/QUMyNzQ2LmpwZz9p/bT1SZXNpemUsd2lk/dGg9NDUw"
        },
        {
            "name": "Men Grey T-Shirt",
            "category": "Clothing",
            "description": "Athletic grey t-shirt by Puma, perfect for sports and workouts",
            "image_url": "https://imgs.search.brave.com/SjYYt4_SGTEpxCEHj-0_K7yFkxPU0S1UPdQ4S5Es9rM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pNS53/YWxtYXJ0aW1hZ2Vz/LmNvbS9zZW8vR2ls/ZGFuLU1lbi1MaWdo/dHdlaWdodC1ULVNo/aXJ0LVNvZnRzdHls/ZS1TaG9ydC1TbGVl/dmUtQ3JhZnRpbmct/VGVlLUhlYXRoZXIt/RGFyay1HcmV5LVNp/emVzLVMtM1hMLTY1/LTM1LVBvbHktQ290/dG9uXzVhNjI2MGVk/LTg5ZWUtNDhlNy1h/OTI3LTMxMDNhYzc2/ZmI5Yi5kOTlmYjk3/ZDg5YzQ5MWIwYzVh/ZDJmNjRmOGZjNzlj/OC5qcGVnP29kbkhl/aWdodD01ODAmb2Ru/V2lkdGg9NTgwJm9k/bkJnPUZGRkZGRg"
        },
        {
            "name": "Blue Denim Jeans",
            "category": "Clothing",
            "description": "Classic slim fit blue denim jeans for men",
            "image_url": "https://imgs.search.brave.com/8hpDCOO--lfhQt3TvqlKhLLP5UWOePFQLXUaS33NnNk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9kZWFy/Ym9ybmRlbmltLnVz/L2Nkbi9zaG9wL3By/b2R1Y3RzLzIwYTk1/MzliYWQyNzc3MGE3/ZDEzMTQ2NmUzYTVi/MDUzLmpwZz92PTE2/MDA5Njc2NjY"
        },
        {
            "name": "Men’s Formal White Shirt",
            "category": "Clothing",
            "description": "Crisp and stylish white shirt for formal occasions",
            "image_url": "https://imgs.search.brave.com/YvUNOQH0c9YnabMmdiJP4ywx_4Ii45cUY9EI_a0Y6rc/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9hc3Nl/dHMubXludGFzc2V0/cy5jb20vZHByXzIs/cV82MCx3XzIxMCxj/X2xpbWl0LGZsX3By/b2dyZXNzaXZlL2Fz/c2V0cy9pbWFnZXMv/MTQ3MjgwMjgvMjAy/MS84LzkvZGJiNzEz/YzMtNDExZS00NjQz/LWJkYTMtZGU0MDNk/MzBhOWFjMTYyODUw/NzU2MzI0Mi1JVk9D/LU1lbi1SZWd1bGFy/LUZpdC1Tb2xpZC1G/b3JtYWwtU2hpcnQt/OTI3MTYyODUwNzU2/MjYyMy0xLmpwZw"
        },
        {
            "name": "Men’s Blue Casual Shirt",
            "category": "Clothing",
            "description": "Comfortable blue casual shirt suitable for outings",
            "image_url": "https://imgs.search.brave.com/k67UXiJarYAnE18oaxeumkKzQXFCWoTLtfW4LmRD110/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NjFuUmdpRmh6OEwu/anBn"
        },
        {
            "name": "Men’s Checked Casual Shirt",
            "category": "Clothing",
            "description": "Trendy checked casual shirt for a stylish look",
            "image_url": "https://imgs.search.brave.com/hgiGWY2YoF8NuZtgdM0tvcVOhw2q3k1bt6yK2OWdXeI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NjFqZXR1aHpraUwu/anBn"
        },
        {
            "name": "Men Black Watch",
            "category": "Accessories",
            "description": "Premium black Skagen analog watch for men",
            "image_url": "https://imgs.search.brave.com/DqcXHKk0SDSpnwJsepVm8r0QkxqsNiEaaau1k6c2NqQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9mb3Nz/aWwuc2NlbmU3LmNv/bS9pcy9pbWFnZS9G/b3NzaWxQYXJ0bmVy/cy9EWjQxODBfbWFp/bj8kc2ZjY19mb3Nf/bWVkaXVtJA"
        },
        {
            "name": "Men Black Sports Shoes",
            "category": "Footwear",
            "description": "Lightweight and durable black sports shoes by Adidas",
            "image_url": "https://imgs.search.brave.com/WlHOje4g9Hox_eiyc_cp1S5oIpt3a3szFnkVfv_oJiU/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWFn/ZXMucHVtYS5jb20v/aW1hZ2UvdXBsb2Fk/L2ZfYXV0byxxX2F1/dG8sYl9yZ2I6ZmFm/YWZhLHdfMzAwLGhf/MzAwL2dsb2JhbC8z/MTE4MTEvMDEvc3Yw/MS9mbmQvSU5EL2Zt/dC9wbmcvU29mdHJp/ZGUtUHJvLUR5bmFt/aWMtRmxleC1NZW4n/cy1TcG9ydHMtU2hv/ZXM"
        },
        {
            "name": "Fossil Men Silver Watch",
            "category": "Accessories",
            "description": "Stylish silver chronograph watch from Fossil",
            "image_url": "https://imgs.search.brave.com/cv8CmlN1UipbdKC2R7YiykmKfDlZhgHanD7fVWbfoPM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/c29uYXRhd2F0Y2hl/cy5pbi9kdy9pbWFn/ZS92Mi9CS0REX1BS/RC9vbi9kZW1hbmR3/YXJlLnN0YXRpYy8t/L1NpdGVzLXRpdGFu/LW1hc3Rlci1jYXRh/bG9nL2RlZmF1bHQv/ZHc0NjQ5YjRiYy9p/bWFnZXMvU29uYXRh/L0NhdGFsb2cvNzk4/N1NNMTBXXzEuanBn/P3N3PTM2MCZzaD0z/NjA"
        },
        {
            "name": "Men White Sneakers",
            "category": "Footwear",
            "description": "Trendy and casual white sneakers from HRX",
            "image_url": "https://imgs.search.brave.com/vsT1P4xlhlub3WmccXiVNtbo7BHSQh1Y8L5aQUhrjPI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NjFEZmZIc3c5REwu/anBn"
        },
        {
            "name": "Allen Solly Men Navy Shirt",
            "category": "Clothing",
            "description": "Formal navy blue shirt by Allen Solly",
            "image_url": "https://imgs.search.brave.com/xgXRjNRINbRhE4q4jQSQFY0NYPuirWiLapVMwSe0At0/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zYXZp/bGVyb3djby5jb20v/Y2RuL3Nob3AvZmls/ZXMvbmF2eS1saW5l/bi1ibGVuZC1jbGFz/c2ljLWZpdC1zaG9y/dC1zbGVldmUtc2hp/cnQtMTM1N25hdm1z/cy5qcGc_dj0xNzU0/NDg4NTUxJndpZHRo/PTQwMA"
        },
        {
            "name": "Brown Leather Wallet",
            "category": "Accessories",
            "description": "Premium brown leather wallet by Hidesign",
            "image_url": "https://imgs.search.brave.com/zxZRfdWyWEoylFSZlmFas0ehhbdEeBwPF6ICxB1_6Rg/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvMTE2/MzE5MjM0My9waG90/by9jbG9zZWQtYnJv/d24tbGVhdGhlci13/YWxsZXQtaXNvbGF0/ZWQtb24tYS13aGl0/ZS1iYWNrZ3JvdW5k/LmpwZz9zPTYxMng2/MTImdz0wJms9MjAm/Yz02RHRnaUpScTcy/al91RlJ4T00zVEhs/ZHAxTm1zbnc2SXhO/MFhqcFBjMUY4PQ"
        },
        {
            "name": "Men Brown Boots",
            "category": "Footwear",
            "description": "Rugged and durable brown boots from Woodland",
            "image_url": "https://imgs.search.brave.com/iPvkalxET0FRcYK7SvzKADMcCC5OaJfX7_ZUsowWryU/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90aHVy/c2RheWJvb3RzLmNv/bS9jZG4vc2hvcC9m/aWxlcy8yODAweDEw/ODAtTWVuLUNhcHRh/aW4tQnJvd24tMjAw/NDE3XzgwMHg1MzNf/Y3JvcF9yaWdodC5q/cGc_dj0xNjEzNzcx/ODU4"
        },
        {
            "name": "Lavie Women Black Handbag",
            "category": "Bags",
            "description": "Spacious and elegant black handbag from Lavie",
            "image_url": "https://imgs.search.brave.com/Lti0KPW2EwVe15aIsZCen2DVgTP9nz6Dg0Z-LCyYeIQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NzErZWZDVVA1Z0wu/anBn"
        },
        {
            "name": "Men Blue Sports Shoes",
            "category": "Footwear",
            "description": "Comfortable blue sports shoes from Fila",
            "image_url": "https://imgs.search.brave.com/hACB95D2f7sgiWLiMAgWoXDDgDK1oxJ0n2VymQQuC0M/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pNS53/YWxtYXJ0aW1hZ2Vz/LmNvbS9zZW8vS3Jp/Y2VseS1NZW4tcy1U/cmFpbC1SdW5uaW5n/LVNob2VzLUZhc2hp/b24tV2Fsa2luZy1I/aWtpbmctU25lYWtl/cnMtTWVuLVRlbm5p/cy1Dcm9zcy1UcmFp/bmluZy1TaG9lLU91/dGRvb3ItU25lYXJr/ZXItTWVucy1DYXN1/YWwtV29ya291dC1G/b290d2Vhci1CbHVl/LVNfODRkOGE5MTgt/NjQ1My00ZWVmLWI5/ZGQtMjdlODk2Y2Fj/ZTRlLmI3N2RlMGQy/NzExODE4NDE2NjUw/NWVjZTM0M2I3NTQ5/LmpwZWc_b2RuSGVp/Z2h0PTU4MCZvZG5X/aWR0aD01ODAmb2Ru/Qmc9RkZGRkZG"
        },
        {
            "name": "Men Grey Blazer",
            "category": "Clothing",
            "description": "Stylish grey formal blazer from Arrow",
            "image_url": "https://imgs.search.brave.com/hboCNGhx1sXZ0_IdOIIG7RRBvkJuxLJM_65X62eK4Mo/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/eHBvc2VkbG9uZG9u/LmNvbS9jZG4vc2hv/cC9maWxlcy8wNDA2/MjRDcmF6ZWQxMzcz/OC5qcGc_dj0xNzIx/Mzg3OTQ2"
        },
        {
            "name": "Men Black Track Pants",
            "category": "Clothing",
            "description": "Comfortable black track pants for training and workouts",
            "image_url": "https://imgs.search.brave.com/BDnNqSgF6cvuOI5lSR98KK2gTv7DHfieJpj3F6XXWJo/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWFn/ZXMucHVtYS5jb20v/aW1hZ2UvdXBsb2Fk/L2ZfYXV0byxxX2F1/dG8sYl9yZ2I6ZmFm/YWZhLHdfMzAwLGhf/MzAwL2dsb2JhbC82/Mjk1ODkvMDEvbW9k/MDEvZm5kL1BOQS9m/bXQvcG5nL1Q3LUFM/V0FZUy1PTi1NZW4n/cy1SZWxheGVkLVRy/YWNrLVBhbnRz"
        },
        {
            "name": "Men Black Sunglasses",
            "category": "Accessories",
            "description": "Classic aviator-style black sunglasses",
            "image_url": "https://imgs.search.brave.com/UFZmmGHou20Y9EN7H9JVxfQlpLXyi3ZcscwlN1Wglhs/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zdW5z/a2kuY29tL2Nkbi9z/aG9wL2ZpbGVzL3N1/bnNraV9wb2xhcml6/ZWRfc3VuZ2xhc3Nl/c19ibGFja19zbGF0/ZV9mZWF0dXJlZC5q/cGc_dj0xNzUyMDc5/NzcyJndpZHRoPTcw/MA"
        },
        {
            "name": "Women Beige Heels",
            "category": "Footwear",
            "description": "Elegant beige high heels",
            "image_url": "https://imgs.search.brave.com/EirYOJOJrHNxlARf49i9F5wJ3DujdDyUj2a_D7YzCxI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NjFDNndoRWE3TUwu/anBn"
        },
        {
            "name": "Classic Black Leather Wallet",
            "category": "Accessories",
            "description": "Sleek black genuine leather wallet",
            "image_url": "https://imgs.search.brave.com/v5ogfkINPe3pDSFsIc0DBc06ckNSjtBSeleYnu9e_xA/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9sb25k/b25hbGxleS5pbi9j/ZG4vc2hvcC9maWxl/cy8xXzUyNmZmMDM2/LTc3MjMtNGM2Ny05/NWZmLTY0ZjYxNDI2/NWEzMy5qcGc_dj0x/NzM5Nzk2MjM4"
        },
        {
            "name": "Men’s Denim Jeans",
            "category": "Apparel",
            "description": "Regular-fit blue denim jeans",
            "image_url": "https://imgs.search.brave.com/8hpDCOO--lfhQt3TvqlKhLLP5UWOePFQLXUaS33NnNk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9kZWFy/Ym9ybmRlbmltLnVz/L2Nkbi9zaG9wL3By/b2R1Y3RzLzIwYTk1/MzliYWQyNzc3MGE3/ZDEzMTQ2NmUzYTVi/MDUzLmpwZz92PTE2/MDA5Njc2NjY"
        },
        {
            "name": "Women Red Evening Gown",
            "category": "Apparel",
            "description": "Floor-length red evening dress",
            "image_url": "https://imgs.search.brave.com/jox2zBYLuu9SNI0m7v_5q_pYX97zLF-ue7W8MtpIQ2g/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLmV0/c3lzdGF0aWMuY29t/LzQ5OTQ5Mjc4L3Iv/aWwvMWEzZDEyLzU4/Mzg1MjczMTYvaWxf/NjAweDYwMC41ODM4/NTI3MzE2X3Q0NnIu/anBn"
        },
        {
            "name": "Bluetooth Wireless Earbuds",
            "category": "Electronics",
            "description": "Compact and powerful wireless earbuds",
            "image_url": "https://imgs.search.brave.com/F50HB254W6J5OL4hqV2Klxp4omN66cWlrGOxtK1J4SI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWFn/ZXMtbmEuc3NsLWlt/YWdlcy1hbWF6b24u/Y29tL2ltYWdlcy9J/LzYxelc4eWM0aFRM/LmpwZw"
        },
        {
            "name": "Gaming Mechanical Keyboard",
            "category": "Electronics",
            "description": "RGB backlit mechanical keyboard",
            "image_url": "https://imgs.search.brave.com/ldh-sXYLyHssH4YD_DbvIENSd5zEzsz7cIDjdMA3JuQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9yZWRy/YWdvbnNob3AuY29t/L2Nkbi9zaG9wL2Zp/bGVzL1JlZHJhZ29u/SzY1NlBSTzMtTW9k/ZTEwMEtleXNXaXJl/bGVzc1JHQkdhbWlu/Z0tleWJvYXJkXzEu/cG5nP3Y9MTc0Njc3/NzEyNCZ3aWR0aD01/MzM"
        },
        {
            "name": "Ceramic Coffee Mug",
            "category": "Home & Kitchen",
            "description": "12oz white ceramic coffee mug",
            "image_url": "https://imgs.search.brave.com/QbBjmDk2hkNZSaUxaUR26Mcg6KHtUeK69-j9LymfGis/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NTFjVHNuWjN3eUwu/anBn"
        },
        {
            "name": "Wooden Cutting Board",
            "category": "Home & Kitchen",
            "description": "Bamboo chopping board, 12x8 in",
            "image_url": "https://imgs.search.brave.com/XOhWLubJxrHtRClenKpncpBrrXHpSMWBVjSwHPHcPYw/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvMTQ4/NzMxMzIwOS9waG90/by93b29kZW4tY3V0/dGluZy1ib2FyZC1v/bi1raXRjaGVuLXRh/YmxlLndlYnA_YT0x/JmI9MSZzPTYxMng2/MTImdz0wJms9MjAm/Yz1ZMWdaTHpjUTNr/WkpSUWZnQU1xUWd0/R3c5UFloY1hYU1A0/ckQwQ3p2dWcwPQ"
        },
        {
            "name": "Stainless Steel Water Bottle",
            "category": "Outdoors",
            "description": "Insulated 500mL water bottle",
            "image_url": "https://imgs.search.brave.com/PTPhH4pUFIaBJhu6We8PlP-Z_yFB1YAy8YpK7JW7zuc/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90My5m/dGNkbi5uZXQvanBn/LzEzLzcwLzkzLzM0/LzM2MF9GXzEzNzA5/MzM0MjJfR3dqSGRO/ODNyeTlhMjdrRW11/bUswRVV4a1AzUWFW/TXcuanBn"
        },
        {
            "name": "Hiking Trail Backpack",
            "category": "Outdoors",
            "description": "30L waterproof hiking backpack",
            "image_url": "https://imgs.search.brave.com/YpkeSgsBK8wDKi-KQdUuB7ZYhpQUKdpA83iSnYYPXDY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9kazBm/a2p5Z2JuOXZ1LmNs/b3VkZnJvbnQubmV0/L2NhY2hlLWJ1c3Rl/ci0xMTc1NTA1OTcz/MS9kZXV0ZXIvbWVk/aWFyb29tL3Byb2R1/Y3QtaW1hZ2VzL2Jh/Y2twYWNrcy9oaWtp/bmctYmFja3BhY2tz/LzE0NTg2MS9pbWFn/ZS10aHVtYl9fMTQ1/ODYxX19kZXV0ZXJf/cHJvZHVjdC10ZWFz/ZXIvMzQwMTMyMS03/NDAzLUZ1dHVyYVBy/bzQwX2JsYWNrX2dy/YXBoaXRlLUQtMDAu/cG5n"
        },
        {
            "name": "Yoga Fitness Mat",
            "category": "Sports & Fitness",
            "description": "Non-slip 6mm yoga mat",
            "image_url": "https://imgs.search.brave.com/lUXxZrmuLC0_eufga68HBj9PvOouLinx0vWhg7sFA9k/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NTFvMEI5UkhUTkwu/anBn"
        },
        {
            "name": "Dumbbell Set",
            "category": "Sports & Fitness",
            "description": "Adjustable dumbbell set",
            "image_url": "https://imgs.search.brave.com/qHdoxZt7GILKAno5M-nsfkb93UYSYCUOIDvxrD6ncV4/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NTErYzhWTkRvdUwu/anBn"
        },
        {
            "name": "Men's Wrist Watch",
            "category": "Accessories",
            "description": "Classic analog stainless steel watch",
            "image_url": "https://imgs.search.brave.com/9Hbxn4kbfqsb57P43TEvQtxSUOj46bZ9JveyzfzYS8g/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NDF2bkw4MEZ5dUwu/anBn"
        },
        {
            "name": "Smart Fitness Tracker",
            "category": "Electronics",
            "description": "Wearable activity and sleep tracker",
            "image_url": "https://imgs.search.brave.com/qQaiOMiH7cOM_7kqnn80nPn5ZNnE3y0VLPwUBmC_ugY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9ydW5t/ZWZpdC5jb20vd3At/Y29udGVudC91cGxv/YWRzLzIwMjUvMDEv/cnVubWVmaXRfZ3Rs/Ml9zbWFydF9maXRu/ZXNzX3RyYWNrZXJf/b3ZlcnZpZXctMi53/ZWJw"
        },
        {
            "name": "Headphones",
            "category": "Electronics",
            "description": "Over-ear active noise cancelling headphones",
            "image_url": "https://imgs.search.brave.com/bLRDiDFUQQQfXFslsSjGBtpQqMsA_AYNWpsYXKLHR0Q/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4u/dGhld2lyZWN1dHRl/ci5jb20vd3AtY29u/dGVudC9tZWRpYS8y/MDI1LzA2L0JFU1Qt/Tk9JU0UtQ0FOQ0VM/TElORy1IRUFEUEhP/TkVTLTgyNTMuanBn"
        },
        {
            "name": "Women Crossbody Bag",
            "category": "Bags",
            "description": "Compact beige crossbody shoulder bag",
            "image_url": "https://imgs.search.brave.com/moGlpfXfm6xL2tovadSNTdybkKnZyPRtjbaCAzVZYbk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4u/c2Frc2ZpZnRoYXZl/bnVlLmNvbS9pcy9p/bWFnZS9zYWtzLzA0/MDAwMjA2NDY5NTRf/QkxBQ0s_d2lkPTM4/MCZoZWk9NTA2JnFs/dD03MCZyZXNNb2Rl/PXNoYXJwMiZvcF91/c209MS4yLDEsNiww"
        },
        {
            "name": "Men Leather Belt",
            "category": "Accessories",
            "description": "Brown genuine leather belt, size L",
            "image_url": "https://imgs.search.brave.com/2EW9hBt9ql5OS5T0FGBDXPqE1z7p82yOZmHKmMXiZU4/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9hc3Nl/dHMubXludGFzc2V0/cy5jb20vZHByXzIs/cV82MCx3XzIxMCxj/X2xpbWl0LGZsX3By/b2dyZXNzaXZlL2Fz/c2V0cy9pbWFnZXMv/MjUzMDQ1NDYvMjAy/My8xMC8yLzYxNWI0/MmM0LWYyYzYtNDI2/ZS05YTIzLTllNWM4/ZGIxMzE2ODE2OTYy/Mjk1MDM4OTVCZWx0/czEuanBn"
        },
        {
            "name": "Toddler Toy Blocks Set",
            "category": "Toys",
            "description": "Colorful wooden blocks (50 pcs)",
            "image_url": "https://imgs.search.brave.com/ciNFkozEW6C9XwI76DvSs3bww_CANuPXtEnUZszxjLY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NjE0dzBlTUpGZ0wu/anBn"
        },
        {
            "name": "LED Desk Lamp",
            "category": "Home & Living",
            "description": "Dimmable LED desk lamp with USB",
            "image_url": "https://imgs.search.brave.com/qnzE0dnWZ0jbK83QasEEJ88k3u_qCnj4VYtACe2D8pY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/ZGVzdGluYXRpb25s/aWdodGluZy5jb20v/aW1hZ2VzL3Byb2R1/Y3RzX2R0bC8xNDMv/UDE0OTcxNDMuYmd-/ZHRsLmpwZw"
        },
        {
            "name": "Men’s White Sneakers",
            "category": "Footwear",
            "description": "Casual white canvas sneakers",
            "image_url": "https://imgs.search.brave.com/TeOtEQquxu6xBkW6nmhokIPMZRjbo3X3np72f88UIhI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzLzcxLzFh/LzRjLzcxMWE0Y2Vm/NmYwZDdjMjQ2ZjE3/MGM3NzU1NjM2YWE1/LmpwZw"
        },
        {
            "name": "Ladies Black Sandals",
            "category": "Footwear",
            "description": "Elegant black strappy sandals",
            "image_url": "https://imgs.search.brave.com/JVrUSMqviRZQMtEsOZASJM8BCgE0witBppCeLJfQ55c/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pNS53/YWxtYXJ0aW1hZ2Vz/LmNvbS9zZW8vT3V0/UHJvLVdvbWVuLXMt/SGlraW5nLVNhbmRh/bHMtT3V0ZG9vci1B/dGhsZXRpYy1DYXN1/YWwtU2FuZGFscy1T/dW1tZXItQmVhY2gt/U2hvZXMtZm9yLUZl/bWFsZS1CbGFja18x/NTRhODkzMC00M2Iz/LTRhNzAtODY2NS02/ZWE1MjVjZTJmY2Eu/MTE5MDYwNTA0NmJm/OTgwZDhiOTE2ZTM3/YTZiMzEzYjMuanBl/Zz9vZG5IZWlnaHQ9/NTgwJm9kbldpZHRo/PTU4MCZvZG5CZz1G/RkZGRkY"
        },
        {
            "name": "Leather Oxford Shoes",
            "category": "Footwear",
            "description": "Classic brown leather oxfords",
            "image_url": "https://imgs.search.brave.com/6tuxdyEIEdLfNm2HYbzVpj6PXx_sBNwVuJsyP6pEuFI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9kMWZ1/ZnZ5NHhhbzZrOS5j/bG91ZGZyb250Lm5l/dC9mZWVkL2ltZy9t/YW5fc2hvZS8zOTcw/MTMvc2lkZV9zbWFs/bC5wbmc"
        },
        {
            "name": "LED TV",
            "category": "Electronics",
            "description": "4K UHD smart LED TV",
            "image_url": "https://imgs.search.brave.com/1EYVj5JpoboudkWs2b6ZS636AEpLEHsk2SHqUdur1Dc/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/c2h1dHRlcnN0b2Nr/LmNvbS9pbWFnZS1w/aG90by9ibGFjay1s/ZWQtdHYtdGVsZXZp/c2lvbi1zY3JlZW4t/MjYwbnctNDUzNjc2/OTY5LmpwZw"
        },
        {
            "name": "Wireless Gaming Mouse",
            "category": "Electronics",
            "description": "Ergonomic RGB wireless gaming mouse",
            "image_url": "https://imgs.search.brave.com/jIWbvseRS7XSOG3AlI-Bh1_MtUm6yltNSO2SKn9_sI0/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9hc3Nl/dC11cy1zdG9yZS5t/c2kuY29tL2ltYWdl/L2NhY2hlL2NhdGFs/b2cvUGRfcGFnZS9H/R0QvTU9VU0UvVkVS/U0FQUk9XL1ZFUlNB/UFJPVy0xLTIyOHgy/MjgucG5n"
        },
        {
            "name": "Portable Bluetooth Speaker",
            "category": "Electronics",
            "description": "Compact waterproof Bluetooth speaker",
            "image_url": "https://imgs.search.brave.com/hdnKFMWSe0HghKjSqHNCR7BvBN6GYhwMKUb6gcFXhDI/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4u/bW9zLmNtcy5mdXR1/cmVjZG4ubmV0L1lm/R1RhanhWVWNSUGpt/RllWclJ2d2kuanBn"
        },
        {
            "name": "Stainless Steel Frying Pan",
            "category": "Home & Kitchen",
            "description": "Non-stick 12-inch frying pan",
            "image_url": "https://imgs.search.brave.com/GmP9kMAHpcq--Q1Gl74cWKBUrgpQkUoH46WQheWFadk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90aHVt/YnMuZHJlYW1zdGlt/ZS5jb20vYi9zdGFp/bmxlc3Mtc3RlZWwt/ZnJ5aW5nLXBhbi0x/ODk2NjY0LmpwZw"
        },
        {
            "name": "Silicone Baking Mat",
            "category": "Home & Kitchen",
            "description": "45x30 cm reusable non-stick baking mat",
            "image_url": "https://imgs.search.brave.com/jYGCzC0h6hWlavnQ9CpqB3cpXFCwPIkb2qiQo0SMh1M/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWFn/ZXMtZXUuc3NsLWlt/YWdlcy1hbWF6b24u/Y29tL2ltYWdlcy9J/LzgxcE8rS1VkemJM/Ll9BQ19VTDMwMF9T/UjMwMCwyMDBfLmpw/Zw"
        },
        {
            "name": "Memory Foam Pillow",
            "category": "Home & Living",
            "description": "Ergonomic memory foam pillow",
            "image_url": "https://imgs.search.brave.com/G1EvOXlm76yolpPjZf2NvH9lmASLhCQaDYb0js1Zum8/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9iZmFz/c2V0LmNvc3Rjby1z/dGF0aWMuY29tL1U0/NDdJSDM1L2FzLzY2/Y2d3cDY2dDg3aGtm/eG1xeDlqcnJrLzE3/OTE0MzktODQ3X18x/P2F1dG89d2VicCZm/b3JtYXQ9anBnJndp/ZHRoPTM1MCZoZWln/aHQ9MzUwJmZpdD1i/b3VuZHMmY2FudmFz/PTM1MCwzNTA"
        },
        {
            "name": "Camping Tent",
            "category": "Outdoors",
            "description": "Small 2-person waterproof camping tent",
            "image_url": "https://imgs.search.brave.com/66EIhprJ7PeoU2BfIxezgvL41qx7BjZJBoOhBa6_BVU/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90aHVt/YnMuZHJlYW1zdGlt/ZS5jb20vYi9jYW1w/aW5nLXRlbnQtZ3Jl/ZW4tZ3Jhc3MtZm9y/cmVzdC00NTgyNTU3/OC5qcGc"
        },
        {
            "name": "Wall Art Print",
            "category": "Home Decor",
            "description": "Abstract art print framed poster",
            "image_url": "https://imgs.search.brave.com/tR9FBdWQqdp0aqQVBhRb4FJvRkRwUxXFgKko4A0c4NY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jYi5z/Y2VuZTcuY29tL2lz/L2ltYWdlL0NyYXRl/L1RoZUpvdXJuZXlJ/c015SG9tZVByaW50/U1NGMjIvJHdlYl9w/bHBfY2FyZCQvMjQw/MjAxMTMxODM3L3Ro/ZS1qb3VybmV5LWlz/LW15LWhvbWUtcHJp/bnQuanBn"
        },
        {
            "name": "Soft Plush Teddy Bear",
            "category": "Toys",
            "description": "Large cuddly teddy bear, 60 cm tall",
            "image_url": "https://imgs.search.brave.com/FjXphQ_bDmn18wUkfSi9r1CpDEB0JUEkTykY-EBbk5g/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/NDFGTDdBWkk0Skwu/anBn"
        },

            { 
              "name": "Women Beige Heels", 
              "category": "Footwear", 
              "description": "Elegant beige high heels",
              "image_url": "https://www.aldoshoes.in/on/demandware.static/-/Sites-aldo_master_catalog/default/dwd597542b/large/stessy2.0270033_1.jpg"
            }
        ]

        for data in products_data:
            self.stdout.write(f"Processing product: {data['name']}")
            try:
                if not data.get('image_url') or not data['image_url'].startswith('http'):
                    self.stdout.write(self.style.WARNING(f"Skipping '{data['name']}' due to invalid URL."))
                    continue
                
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(data['image_url'], headers=headers, timeout=15)
                response.raise_for_status()
                image_bytes = response.content

                pil_image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
                features = extract_features(pil_image)

                img_file = ContentFile(image_bytes)
                
                category_name = data.get('category', 'uncategorized')
                filename_base = data.get('name', 'unnamed-product')
                
                safe_category = re.sub(r'[^\w\s-]', '', category_name).strip().replace(' ', '_')
                safe_filename = re.sub(r'[^\w\s-]', '', filename_base).strip().replace(' ', '_')

                if not safe_category: safe_category = "uncategorized"
                if not safe_filename: safe_filename = "product-" + str(np.random.randint(1000, 9999))
                
                img_path = f"{safe_category}/{safe_filename}.jpg"
                
                #self.stdout.write(self.style.WARNING(f"Generated S3 Path: {img_path}"))
                
                product = Product(name=data['name'], category=data['category'])
                
                product.image.save(img_path, img_file, save=False)

                if features is not None:
                    product.feature_vector = features.tobytes()
                
                product.save()
                self.stdout.write(self.style.SUCCESS(f"Successfully saved '{data['name']}' to S3 path: {img_path}"))

            except requests.exceptions.RequestException as e:
                self.stdout.write(self.style.ERROR(f"Could not download image for '{data['name']}': {e}"))
            except PILImage.UnidentifiedImageError:
                 self.stdout.write(self.style.ERROR(f"Could not identify image file for '{data['name']}'. The URL may not be a direct image link."))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"An unexpected error occurred for '{data['name']}': {e}"))
                self.stdout.write(traceback.format_exc())

        self.stdout.write(self.style.SUCCESS("Finished populating all products."))

